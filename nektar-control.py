#!/usr/bin/env python3

"""Control script for Nektar floor unit."""

# this is a Python rewrite of the MIDI Lab script I was using to
# control my Nektar floor unit. the system is entirely ad hoc but
# hopefully it is tractable enough to understand how it works later

from enum import Enum, IntEnum, auto
from queue import Queue
from typing import Optional, Callable, List, Dict, Set
import midi
import rtmidi
from yaml import load as yaml_load
from json import load as js_load
from jsonschema import validate

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# stolen from here:
# https://stackoverflow.com/questions/41134365/formula-to-create-simple-midi-velocity-curves


def _adjust_midi_value(val: int, curve: float) -> int:
    assert(0 <= val <= 127)
    assert(0.0 <= curve <= 1.0)

    # original code had deviation at -100..100, so adjust
    curve = (200.0 * curve) - 100.0

    _max, _mid = 127.0, 63.5

    cpX = _mid + ((curve / 100) * _mid)
    t = float(val) / _max
    d = round(2 * (1 - t) * t * cpX, 0) + (t * t * _max)
    return int((val - d) + val)


class FunctionType(Enum):
    """Action to take on state change."""

    COMMAND = auto()   # sends MIDI CC to DAW
    PRESET = auto()    # switches presets
    SYNC = auto()      # syncs preset to DAW

    @staticmethod
    def from_str(s: str) -> 'FunctionType':
        """Convert string to function type."""
        strmap = {
            "command": FunctionType.COMMAND,
            "preset": FunctionType.PRESET,
            "sync": FunctionType.SYNC,
        }
        s = s.lower()
        try:
            return strmap[s]
        except KeyError as e:
            raise KeyError("Unknown function type") from e

    def __str__(self) -> str:
        """Return string representation of function type."""
        m = {
            FunctionType.COMMAND: "Command",
            FunctionType.PRESET: "Preset",
            FunctionType.SYNC: "Sync",
        }
        return m[self]


class ControlType(Enum):
    """Describes available control types."""

    TOGGLE = auto()    # state change on press
    MOMENTARY = auto()  # state change on press or release
    TRIGGER = auto()   # no state change

    @staticmethod
    def from_str(s: str) -> 'ControlType':
        """Convert a string to a control type."""
        strmap = {
            "toggle": ControlType.TOGGLE,
            "momentary": ControlType.MOMENTARY,
            "trigger": ControlType.TRIGGER
        }
        s = s.lower()
        try:
            return strmap[s]
        except KeyError as e:
            raise KeyError("Unknown control type") from e

    def __str__(self) -> str:
        """Return string representation of control type."""
        m = {
            ControlType.TOGGLE: "Toggle",
            ControlType.MOMENTARY: "Momentary",
            ControlType.TRIGGER: "Trigger",
        }
        return m[self]


class ControlId(IntEnum):
    """Definitions for buttons."""

    BUTTON1 = 0
    BUTTON2 = 1
    BUTTON3 = 2
    BUTTON4 = 3
    BUTTON5 = 4
    BUTTON6 = 5
    BUTTONA = 6
    BUTTONB = 7
    BUTTONC = 8
    BUTTOND = 9
    PEDAL1 = 10

    @staticmethod
    def from_str(s: str) -> 'ControlId':
        """Convert string to Control Id."""
        strmap = {
            "1": ControlId.BUTTON1,
            "2": ControlId.BUTTON2,
            "3": ControlId.BUTTON3,
            "4": ControlId.BUTTON4,
            "5": ControlId.BUTTON5,
            "6": ControlId.BUTTON6,
            "A": ControlId.BUTTONA,
            "B": ControlId.BUTTONB,
            "C": ControlId.BUTTONC,
            "D": ControlId.BUTTOND,
        }
        s = s.upper()
        try:
            return strmap[s]
        except KeyError as e:
            raise KeyError("Unknown control id") from e

    @staticmethod
    def from_cc(cc: int) -> 'ControlId':
        """Convert CC number to Control Id."""
        ccmap = {i: b for i, b in zip(range(len(ControlId)), ControlId)}

        return ccmap[cc]

    def to_cc(self) -> int:
        """Convert Control Id to CC value."""
        return int(self)

    def __str__(self) -> str:
        """Return string representation of button id."""
        m = {
            ControlId.BUTTON1: "1",
            ControlId.BUTTON2: "2",
            ControlId.BUTTON3: "3",
            ControlId.BUTTON4: "4",
            ControlId.BUTTON5: "5",
            ControlId.BUTTON6: "6",
            ControlId.BUTTONA: "A",
            ControlId.BUTTONB: "B",
            ControlId.BUTTONC: "C",
            ControlId.BUTTOND: "D",
            ControlId.PEDAL1: "Pedal",
        }
        return m[self]


class ButtonState(Enum):
    """Definitions for states a button can be in."""

    OFF = auto()
    ON = auto()

    def flip(self) -> 'ButtonState':
        """Return opposite state."""
        if self == ButtonState.ON:
            return ButtonState.OFF
        else:
            return ButtonState.ON

    def __str__(self) -> str:
        """Return string representation of button state."""
        m = {
            ButtonState.ON: "On",
            ButtonState.OFF: "Off",
        }
        return m[self]


class ControlState(Enum):
    """Definitions for states a control can be in."""

    OFF = auto()
    ON = auto()

    def flip(self) -> 'ControlState':
        """Return opposite state."""
        if self == ControlState.ON:
            return ControlState.OFF
        else:
            return ControlState.ON

    @staticmethod
    def from_button(state: ButtonState) -> 'ControlState':
        """Return state corresponding to button state."""
        if state == ButtonState.OFF:
            return ControlState.OFF
        if state == ButtonState.ON:
            return ControlState.ON
        raise ValueError("Unknown button state")

    def __str__(self) -> str:
        """Return string representation of control state."""
        m = {
            ControlState.ON: "On",
            ControlState.OFF: "Off",
        }
        return m[self]


class Preset(Enum):
    """Definitions for presets."""

    RHYTHM = auto()
    SOLO = auto()

    def flip(self) -> 'Preset':
        """Return opposite preset."""
        if self == Preset.RHYTHM:
            return Preset.SOLO
        else:
            return Preset.RHYTHM

    def __str__(self) -> str:
        """Return string representation of preset."""
        m = {
            Preset.RHYTHM: "Rhythm",
            Preset.SOLO: "Solo",
        }
        return m[self]


class Event(object):
    """
    Generic event class.

    Objects of this class are used to communicate between different parts
    of the script (we don't want to depend on specific MIDI CC values, so
    we need need an abstraction - this is probably overengineered but who
    cares).
    """

    def __str__(self) -> str:
        """Stub for string representation."""
        raise NotImplementedError()


class MIDIEvent(Event):
    """
    MIDI event.

    This is what we use to communicate with MIDI ports.
    """

    def __init__(self, msg: midi.Message):
        """Construct a MIDI event."""
        super().__init__()
        assert(isinstance(msg, midi.Message))
        self.msg = msg

    def __str__(self) -> str:
        """Return string representation of MIDI event."""
        return f"MIDI event: {self.msg}"


class HardwareEvent(Event):
    """
    Hardware event class.

    These are events coming from the controller.
    """

    pass


class ButtonEvent(HardwareEvent):
    """
    Button press/release event.

    This event is emitted when we press a button.
    """

    def __init__(self, control: 'Control', state: 'ButtonState'):
        """Button event constructor."""
        super().__init__()
        assert(isinstance(control, Control))
        assert(isinstance(state, ButtonState))
        self.control = control
        self.state = state

    def __str__(self) -> str:
        """Return string representation of button event."""
        page = self.control.page
        evt_name = f'{page.name}:{self.control.name}'
        return f"Button event: {evt_name} {self.state}"


class CommandEvent(Event):
    """
    Command event class.

    These are events coming into the controller logic.
    """

    pass


class PedalEvent(CommandEvent):
    """
    Pedal event.

    This event is emitted when we move the pedal.
    """

    def __init__(self, val: int):
        """Construct pedal event."""
        super().__init__()
        assert(isinstance(val, int))
        self.val = val

    def __str__(self) -> str:
        """Return string representation of pedal event."""
        return f"Pedal event: {self.val}"


class ControlEvent(CommandEvent):
    """
    Command event.

    Button presses are signals from physical buttons, while
    control events are logical signals for the controller.
    """

    def __init__(self, control: 'Control', state: 'ControlState'):
        """Command event constructor."""
        assert(isinstance(control, Control))
        assert(isinstance(state, ControlState))
        self.control = control
        self.state = state

    def __str__(self) -> str:
        """Return string representation of control event."""
        page = self.control.page
        evt_name = f'{page.name}:{self.control.name}'
        return f"Control event: {evt_name} {self.state}"


class PageEvent(CommandEvent):
    """
    Page switch event.

    This event is emitted when we switch to a different page.
    """

    def __init__(self, page: 'Page'):
        """Page event constructor."""
        super().__init__()
        self.page = page

    def __str__(self) -> str:
        """Return string representation of page event."""
        return f"Page event: {self.page.name}"


class PresetEvent(CommandEvent):
    """
    Preset event.

    This event is emitted when we change a preset.
    """

    def __init__(self, preset: 'Preset'):
        """Construct a preset event."""
        super().__init__()
        self.preset = preset

    def __str__(self) -> str:
        """Return string representation of preset event."""
        return f"Preset event: {self.preset}"


class SyncEvent(CommandEvent):
    """
    Sync event.

    This event is emitted when we request sync.
    """

    def __init__(self):
        """Construct a preset event."""
        super().__init__()

    def __str__(self) -> str:
        """Return string representation of preset event."""
        return f"Sync event"


class FeedbackEvent(Event):
    """
    Feedback event.

    This event is emitted when we need to update the status of the
    controller, for example when button is changing state, or when
    we change page.
    """

    def __init__(self, id: 'ControlId', state: 'ControlState'):
        """Feedback event constructor."""
        super().__init__()
        self.id = id
        self.state = state

    def __str__(self) -> str:
        """Return string representation of preset event."""
        return f"Feedback event: {self.id} {self.state}"


class PipelineNodeType(Enum):
    """Definitions for pipeline node types."""

    UNKNOWN = auto()
    SOURCE = auto()
    SINK = auto()
    INTERMEDIARY = auto()


class Control(object):
    """Represents a single button."""

    def __init__(self, name: str, _type: ControlType, _func: FunctionType,
                 button: ControlId, pedal: bool, curve: float,
                 onpress: List[int], onrelease: List[int]):
        """Create Control object."""
        self.name = name
        self.pedal = pedal
        self.type = _type
        self.func = _func
        self.id = button
        self.page: 'Page'  # to be filled later
        self.zone: Optional['Zone'] = None  # to be filled later
        self.state = ControlState.OFF if _type != ControlType.TRIGGER else ControlState.ON
        self.curve = curve
        self.onpress = onpress
        self.onrelease = onrelease


class Page(object):
    """Represents a controller page."""

    def __init__(self, name: str, id: int, buttons: List[Control]):
        """Create Page object."""
        self.name = name
        self.id = id
        self.controls = buttons
        # populate each button's page
        for b in buttons:
            assert(not hasattr(b, 'page'))
            b.page = self


class Zone(object):
    """Represents a controller zone."""

    def __init__(self, name: str, xor: bool, buttons: List[Control]):
        """Construct Zone object."""
        self.name = name
        self.xor = xor
        self.buttons = buttons
        # populate each button's zone
        for b in buttons:
            assert(not b.zone)
            b.zone = self

        # if this is a xor zone, enable first button
        if self.xor:
            first = True
            for b in self.buttons:
                b.state = ControlState.ON if first else ControlState.OFF
                first = False


class Pipeline(object):
    """
    Implements pipeline.

    This will hold the pipeline ADG and forward events from one
    node to the next.
    """

    def __init__(self, verbose: bool = False):
        """Construct a pipeline."""
        self.edges: List['PipelineEdge'] = []
        self.node_type: Dict['PipelineNode', 'PipelineNodeType'] = {}
        self.source: Set['PipelineNode'] = set()
        self.verbose = verbose

    def connect(self, src: 'PipelineNode', dst: 'PipelineNode',
                cmp: Callable[['Event'], bool] = None):
        """Connect two nodes."""
        # are we creating any loggers?
        if not self.verbose:
            self._connect(src, dst, cmp)
            return
        # we are creating loggers, so add them automatically
        name = f"{src.name}->{dst.name}"
        logger = LoggerNode(name)
        self._connect(src, logger, cmp)
        # no need for comparator after logger
        self._connect(logger, dst, None)

    def _connect(self, src: 'PipelineNode', dst: 'PipelineNode',
                 cmp: Callable[['Event'], bool] = None):
        edge = PipelineEdge(src, dst, cmp)
        self.edges.append(edge)

        # resolve node types
        try:
            src_t = self.node_type[src]
        except KeyError:
            src_t = PipelineNodeType.UNKNOWN
        if src_t == PipelineNodeType.UNKNOWN:
            # first time we're seeing this
            src_t = PipelineNodeType.SOURCE
        elif src_t == PipelineNodeType.SINK:
            # this was someone else's end
            src_t = PipelineNodeType.INTERMEDIARY

        try:
            dst_t = self.node_type[dst]
        except KeyError:
            dst_t = PipelineNodeType.UNKNOWN
        if dst_t == PipelineNodeType.UNKNOWN:
            # first time we're seeing this
            dst_t = PipelineNodeType.SINK
        elif dst_t == PipelineNodeType.SOURCE:
            # this was someone else's beginning
            dst_t = PipelineNodeType.INTERMEDIARY

        # update node types
        self.node_type[src] = src_t
        self.node_type[dst] = dst_t

        # update list of source nodes
        try:
            if src_t == PipelineNodeType.SOURCE:
                self.source.add(src)
            else:
                self.source.remove(src)
        except KeyError:
            pass
        # destination node by definition cannot be source, so no need to check
        try:
            self.source.remove(dst)
        except KeyError:
            pass

    def run(self):
        """Run the pipeline."""
        # start with the source nodes
        pipeline_queue: Queue[PipelineNode] = Queue()
        for n in self.source:
            pipeline_queue.put(n)

        first = True

        while not pipeline_queue.empty():
            # get next node
            n = pipeline_queue.get()
            # process the node
            n.process()
            # find next nodes
            edges = list(filter(lambda x: x.src == n, self.edges))
            # copy stuff from node's outqueue to next node's inqueue
            while not n.outqueue.empty():
                if first:
                    first = False
                ev = n.outqueue.get()
                for e in edges:
                    # check if filter passes
                    if e.cmp(ev):
                        e.dst.enqueue(ev)
                    elif self.verbose:
                        print(f"{e.dst.name}: Event {ev} dropped")
                    # process next node at next pipeline iteration
                    pipeline_queue.put(e.dst)
        if not first and self.verbose:
            print()


class PipelineEdge(object):
    """Defines a connection between two pipeline nodes."""

    def __init__(self, src: 'PipelineNode', dst: 'PipelineNode',
                 cmp: Optional[Callable[['Event'], bool]] = None):
        """Construct a pipeline edge connection."""
        self.src = src
        self.dst = dst
        if not cmp:
            def _cmp(_): return True
            cmp = _cmp
        self.cmp = cmp


class PipelineNode(object):
    """
    Generic event pipeline node.

    We will be building pipeline of events for each stage of the
    processing pipeline. This is base class for them.
    """

    def __init__(self, name: str):
        """Construct a pipeline node."""
        self.inqueue: Queue['Event'] = Queue()  # input queue
        self.outqueue: Queue['Event'] = Queue()  # output queue
        self.name = name

    def enqueue(self, event: 'Event'):
        """Enqueue an event to the pipeline."""
        self.inqueue.put(event)

    def _accept(self, event: 'Event'):
        self.outqueue.put(event)

    def process(self):
        """Process the pipeline."""
        while not self.inqueue.empty():
            event = self.inqueue.get()
            self._process(event)

    def _process(self, _: 'Event'):
        raise NotImplementedError()


class LoggerNode(PipelineNode):
    """Logger node to dump events passing through."""

    def __init__(self, name: str):
        """Construct a logger node."""
        super().__init__(name)

    def _process(self, event: 'Event'):
        print(f"{self.name}: {event}")
        # pass the event unchanged
        self._accept(event)


class MIDIInputNode(PipelineNode):
    """
    MIDI Input pipeline node.

    Reads MIDI input into the pipeline.
    """

    def __init__(self, name: str, midiin: rtmidi.MidiIn):
        """Construct a MIDI input node."""
        super().__init__(name)
        self._midiin: rtmidi.MidiIn = midiin

    # override process because we don't need the loop
    def process(self):
        """Process MIDI input."""
        assert(self.inqueue.empty())
        while True:
            msg_t = self._midiin.get_message()
            if not msg_t:
                break
            msg, _ = msg_t
            try:
                ccmsg = midi.midi.build_message_from_sequence(msg)
            except AssertionError:
                # this is probably a SysEx
                assert(len(msg) == 6)
                m_id = msg[1]  # extract manufacturer ID
                msg = msg[2:-1]  # remove header and footer
                ccmsg = midi.Message(midi.SysEx(m_id, *msg))
            event = MIDIEvent(ccmsg)
            self._accept(event)


class DecoderNode(PipelineNode):
    """Decoder from MIDI messages to commands."""

    def __init__(self, name: str):
        """Construct a decoder node."""
        super().__init__(name)

    def _process(self, event: 'Event'):
        """Decode enqueued messages."""
        assert(isinstance(event, MIDIEvent))

        # is this a CC message?
        msg = event.msg
        if isinstance(msg.type, midi.ControlChange):
            cc = msg.type._data1
            val = msg.type._data2

            btnid = ControlId.from_cc(cc)

            # is this a button or a pedal?
            if btnid == ControlId.PEDAL1:
                event = PedalEvent(val)
            else:
                try:
                    btn = _global_state.find_button_on_current_page(btnid)
                except IndexError as e:
                    # didn't find this button, ignore
                    return
                state = ButtonState.ON if val == 127 else ButtonState.OFF
                event = ButtonEvent(btn, state)
        elif isinstance(msg.type, midi.SysEx):
            # this is a page switch event
            dev_id = msg.data[0]
            cmd = msg.data[1]
            cmd_id = msg.data[2]

            # page select
            if dev_id == 0 and cmd == 6:
                try:
                    page = _global_state.find_page_by_id(cmd_id)
                except IndexError as e:
                    raise ValueError("Invalid page ID") from e
                event = PageEvent(page)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        self._accept(event)


class TranslatorNode(PipelineNode):
    """Some events can be compound, this splits them up."""

    def __init__(self, name: str):
        """Construct a translator node."""
        super().__init__(name)

    def _switch_preset(self):
        cur_preset = _global_state.preset
        new_preset = cur_preset.flip()

        for b, vals in _global_state.preset_values.items():
            old_val = b.state
            new_val = vals[new_preset]
            vals[cur_preset] = old_val
            if old_val == new_val:
                # no change, skip
                continue
            event = ControlEvent(b, new_val)
            self._accept(event)
        # switch preset too
        pevent = PresetEvent(new_preset)
        self._accept(pevent)

    def _handle_zones(self, event: 'ControlEvent'):
        btn = event.control
        # buttons can be part of zones, in which case special rules apply
        if not btn.zone:
            # no zones here, pass through
            self._accept(event)
            return
        # this button is part of a zone, so let's see if we need to do
        # anything else here
        state = event.state
        if state == ControlState.OFF and btn.zone.xor:
            # xor zones need at least one button active, so if we're
            # turning something off, we shouldn't be able to
            return
        if state == ControlState.ON:
            # we need to turn off other buttons first
            zone = btn.zone
            for b in zone.buttons:
                if b == btn:
                    continue
                if b.state == ControlState.OFF:
                    continue
                zoneevent = ControlEvent(b, ControlState.OFF)
                self._accept(zoneevent)
        # pass on the original event
        self._accept(event)

    def _process_control(self, event: 'ButtonEvent'):
        # not all events can reach buttons, we only need events
        # that can potentially trigger feedback or MIDI messages
        btn = event.control
        # state of physical button
        btn_state = event.state
        # command state
        cmd_state = btn.state
        # if we pass the trigger, this will be the new command state
        new_state = cmd_state.flip()

        if btn.type == ControlType.TOGGLE:
            # if this button is a toggle, ignore "off" messages
            if btn_state == ButtonState.OFF:
                return
            # we are transitioning state, so toggle button state
            cevent = ControlEvent(btn, new_state)
            self._handle_zones(cevent)
        elif btn.type == ControlType.MOMENTARY:
            # translate button state to control state
            new_state = ControlState.from_button(btn_state)
            cevent = ControlEvent(btn, new_state)
            self._handle_zones(cevent)
        elif btn.type == ControlType.TRIGGER:
            # if this is a trigger button, ignore "off" messages
            if btn_state == ButtonState.OFF:
                return
            # "trigger" means we always send the "on" state
            cevent = ControlEvent(btn, ControlState.ON)
            self._handle_zones(cevent)
        else:
            raise NotImplementedError()

    def _process(self, event: 'Event'):
        assert(isinstance(event, HardwareEvent)
               or isinstance(event, CommandEvent))
        if isinstance(event, ButtonEvent):
            # button presses are driving logical commands, so translate
            if event.control.func == FunctionType.PRESET:
                # presets don't need an "off" toggle
                if event.state == ButtonState.OFF:
                    return
                self._switch_preset()
            elif event.control.func == FunctionType.SYNC:
                # sync doesn't need an "off" toggle
                if event.state == ButtonState.OFF:
                    return
                # convert into sync event and pass through
                evt = SyncEvent()
                self._accept(evt)
            elif event.control.func == FunctionType.COMMAND:
                self._process_control(event)
            else:
                raise NotImplementedError()
        elif isinstance(event, CommandEvent):
            # pass through
            self._accept(event)
        else:
            raise NotImplementedError()


class ProcessorNode(PipelineNode):
    """This will react to events and possibly produce more of them."""

    def __init__(self, name: str):
        """Construct a processor node."""
        super().__init__(name)

    def _generate_midi(self, c: 'Control', state: ControlState) -> List[MIDIEvent]:
        assert(c.func == FunctionType.COMMAND)
        cc = self._ctrl_to_cc(c)

        # read CC values from config
        events: List['MIDIEvent'] = []
        lst = c.onpress if state == ControlState.ON else c.onrelease
        for cv in lst:
            msg = midi.Message(midi.ControlChange(cc, cv), 1)
            evt = MIDIEvent(msg)
            events.append(evt)
        return events

    def _generate_pedal_midi(self, c: 'Control', val: int) -> MIDIEvent:
        assert(c.func == FunctionType.COMMAND)
        assert(c.pedal)
        cc = self._ctrl_to_cc(c)

        # adjust pedal response
        val = _adjust_midi_value(val, c.curve)

        msg = midi.Message(midi.ControlChange(cc, val), 2)
        return MIDIEvent(msg)

    def _ctrl_to_cc(self, c: 'Control') -> int:
        page = c.page
        base_cc = page.id * 10
        cc_offset = c.id.to_cc()
        cc = base_cc + cc_offset + 1
        return cc

    def _process_control(self, event: 'ControlEvent'):
        ctrl = event.control
        state = event.state
        ctrl.state = state

        # is this a pedal button?
        if ctrl.pedal:
            if state == ControlState.ON:
                # add it to the list of active pedals
                _global_state.active_pedals.append(ctrl)
                # generate MIDI signal for current position
                mevt = self._generate_pedal_midi(ctrl, _global_state.pedal_pos)
                self._accept(mevt)
            elif state == ControlState.OFF:
                # remove it from list of active pedals
                _global_state.active_pedals.remove(ctrl)
            else:
                raise NotImplementedError()

        # generate MIDI signal
        evts = self._generate_midi(ctrl, state)
        for e in evts:
            self._accept(e)

        # generate feedback if on current page
        if ctrl.page == _global_state.active_page:
            f = FeedbackEvent(ctrl.id, event.state)
            self._accept(f)

    def _process_page(self, event: 'PageEvent'):
        p = event.page
        _global_state.active_page = p

        # generate feedback events on page switch
        for id in ControlId:
            try:
                btn = _global_state.find_button_on_current_page(id)
                # button can be a preset, this is a special case
                if btn.func == FunctionType.PRESET:
                    state = ControlState.ON if _global_state.preset == Preset.SOLO else ControlState.OFF
                else:
                    state = btn.state
            except IndexError:
                state = ControlState.OFF
            fev = FeedbackEvent(id, state)
            # push the event further down the pipe
            self._accept(fev)

    def _process_preset(self, event: 'PresetEvent'):
        _global_state.preset = event.preset
        # find preset button on current page
        try:
            cur_page = _global_state.active_page
            btns = list(filter(lambda x: x.func ==
                        FunctionType.PRESET, cur_page.controls))
            btn = btns[0]
            state = ControlState.ON if event.preset == Preset.SOLO else ControlState.OFF
            f = FeedbackEvent(btn.id, state)
            self._accept(f)
        except IndexError:
            # no preset buttons here, ignore
            pass

    def _process_pedal(self, event: 'PedalEvent'):
        # store last known pedal position
        _global_state.pedal_pos = event.val

        for c in _global_state.active_pedals:
            evt = self._generate_pedal_midi(c, event.val)
            self._accept(evt)

    def _trigger_sync(self):
        # sync controls
        for c in _global_state.preset_values.keys():
            cur_state = c.state
            evts = self._generate_midi(c, cur_state)
            for e in evts:
                self._accept(e)
        # sync pedals
        for c in _global_state.active_pedals:
            evt = self._generate_pedal_midi(c, _global_state.pedal_pos)
            self._accept(evt)
        # sync feedback by triggering page switch
        self._process_page(PageEvent(_global_state.active_page))

    def _process(self, event: 'Event'):
        assert(isinstance(event, CommandEvent))

        if isinstance(event, PageEvent):
            self._process_page(event)
        elif isinstance(event, ControlEvent):
            self._process_control(event)
        elif isinstance(event, PresetEvent):
            self._process_preset(event)
        elif isinstance(event, PedalEvent):
            self._process_pedal(event)
        elif isinstance(event, SyncEvent):
            self._trigger_sync()
        else:
            raise NotImplementedError()


class EncoderNode(PipelineNode):
    """Encoder to MIDI messages from feedback events."""

    def __init__(self, name: str):
        """Construct a encoder node."""
        super().__init__(name)

    def _process(self, event: 'Event'):
        assert(isinstance(event, MIDIEvent)
               or isinstance(event, FeedbackEvent))
        if isinstance(event, MIDIEvent):
            # pass through MIDI events
            pass
        elif isinstance(event, FeedbackEvent):
            # turn feedback events into MIDI events
            cc = event.id.to_cc()
            state = event.state
            val = 127 if state == ControlState.ON else 0
            ccmsg = midi.Message(midi.ControlChange(cc, val), 1)
            event = MIDIEvent(ccmsg)
        else:
            raise NotImplementedError()
        self._accept(event)


class MIDIOutputNode(PipelineNode):
    """
    MIDI output pipeline node.

    Write MIDI messages into MIDI outputs.
    """

    def __init__(self, name: str, midiout: rtmidi.MidiOut):
        """Construct a MIDI output node."""
        super().__init__(name)
        self._midiout = midiout

    def _process(self, event: 'Event'):
        """Process MIDI output."""
        assert(isinstance(event, MIDIEvent))
        self._midiout.send_message(event.msg)


class GlobalState(object):
    """Global state of the controller."""

    def __init__(self):
        """Construct global state."""
        self.active_pedals: List['Control'] = []
        self.active_page: Page
        self.preset = Preset.RHYTHM
        self.pages: List['Page'] = []
        self.zones: List['Zone'] = []
        self.preset_values: Dict['Control', Dict[Preset, ControlState]] = {}
        self.pedal_pos: int = 0
        self.midi_in: str
        self.midi_out: str
        self.fb_out: str

    def find_page_by_id(self, id: int) -> 'Page':
        """Find page by id."""
        p: List['Page'] = list(filter(lambda x: x.id == id, self.pages))
        return p[0]

    def find_button_on_current_page(self, id: ControlId) -> 'Control':
        """Find button by its index."""
        # find active page
        p = self.active_page
        # find button on that page
        btns = list(filter(lambda x: x.id == id, p.controls))
        return btns[0]


_global_state = GlobalState()


def _read_config():
    with open("schema.json") as f:
        schema = js_load(f)
    with open("config.yaml") as f:
        config = yaml_load(f, Loader)
    validate(config, schema)

    _global_state.fb_out = config["midiconfig"]["feedback"]
    _global_state.midi_in = config["midiconfig"]["input"]
    _global_state.midi_out = config["midiconfig"]["output"]

    btncache: Dict[str, Control] = {}

    def _get_default(d: dict, p: str):
        return d[p]["default"]

    def _set_defaults(base_def: dict, obj: dict):
        properties = base_def.keys()
        for p in properties:
            try:
                dv = _get_default(base_def, p)
                obj.setdefault(p, dv)
            except KeyError:
                pass
    for p in config["pages"]:
        name = p["name"]
        id = p["id"]
        buttons = []
        for b in p["controls"]:
            # set default values first, then parse
            b_base_def = schema["$defs"]["control"]["properties"]
            _set_defaults(b_base_def, b)

            bname = b["name"]
            btype = ControlType.from_str(b["type"])
            bfunc = FunctionType.from_str(b["function"])
            bid = ControlId.from_str(b["button"])
            curve = b['curve']
            pedal = b['pedal']
            onpress = b['onpress']
            onrelease = b['onrelease']

            b_obj = Control(bname, btype, bfunc, bid, pedal, curve,
                            onpress, onrelease)

            buttons.append(b_obj)

            # we need these later
            path = f'{name}:{bname}'
            assert(path not in btncache)
            btncache[path] = b_obj

        page = Page(name, id, buttons)
        _global_state.pages.append(page)

    # we have all pages and buttons now, so populate the zones
    for z in config["zones"]:
        # set default values first, then parse
        z_base_def = schema["$defs"]["zone"]["properties"]
        _set_defaults(z_base_def, b)
        name = z["name"]
        try:
            xor = z["xor"]
        except KeyError:
            xor = False
        buttons = [btncache[b] for b in z["controls"]]

        zone = Zone(name, xor, buttons)
        _global_state.zones.append(zone)

    # populate preset structures
    for pb in config["preset"]:
        btn = btncache[pb]
        _global_state.preset_values[btn] = {p: btn.state for p in Preset}

    # select current page
    _global_state.active_page = _global_state.pages[0]


def __main():
    _read_config()

    # create MIDI port handles
    midiin = rtmidi.MidiIn()
    fbout = rtmidi.MidiOut()
    midiout = rtmidi.MidiOut()

    # find port strings
    out_ports = [s[:-2] for s in fbout.get_ports()]
    in_ports = [s[:-2] for s in midiin.get_ports()]

    # MIDI input from controller
    pacer_in = in_ports.index(_global_state.midi_in)
    midiin.open_port(pacer_in)
    midiin.ignore_types(sysex=False, timing=True, active_sense=True)

    # Feedback MIDI output back to the controller
    fb_out = out_ports.index(_global_state.fb_out)
    fbout.open_port(fb_out)

    # MIDI output to DAW
    midi_out = out_ports.index(_global_state.midi_out)
    midiout.open_port(midi_out)

    # create pipeline nodes
    innode = MIDIInputNode("MIDI In", midiin)
    decoder = DecoderNode("Decoder")
    translator = TranslatorNode("Translator")
    processor = ProcessorNode("Processor")
    fbencoder = EncoderNode("Feedback Encoder")
    outencoder = EncoderNode("Output Encoder")
    fbnode = MIDIOutputNode("Feedback Out", fbout)
    outnode = MIDIOutputNode("MIDI Out", midiout)

    # filter out everything but feedback
    def fbfilter(e: 'Event') -> bool:
        return isinstance(e, FeedbackEvent)
    # filter out everything but MIDI events

    def outfilter(e: 'Event') -> bool:
        return isinstance(e, MIDIEvent)

    pipeline = Pipeline(verbose=False)
    # input -> decoder
    pipeline.connect(innode, decoder)
    # decoder -> translator
    pipeline.connect(decoder, translator)
    # traslator -> processor
    pipeline.connect(translator, processor)
    # processor -> feedback encoder -> feedback out
    pipeline.connect(processor, fbencoder, fbfilter)
    pipeline.connect(fbencoder, fbnode)
    # processor -> midi encoder -> midi out
    pipeline.connect(processor, outencoder, outfilter)
    pipeline.connect(outencoder, outnode)

    while True:
        pipeline.run()


if __name__ == '__main__':
    __main()
