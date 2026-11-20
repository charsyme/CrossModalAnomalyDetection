"""ROS-independent checks for keyboard/GUI fault hold state transitions.

Loads only controller methods (not the ROS / Tk initializers), allowing this
regression test to run on development hosts without a ROS 2 installation.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / (
    'src/sensor_fault_simulation/sensor_fault_simulation/fault_controller.py')
source_tree = ast.parse(SOURCE.read_text())
original = next(n for n in source_tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'KeyboardController')
methods = {'active', 'on_press', 'on_release', 'finish_release',
           'on_mouse_press', 'on_mouse_release', 'publish_once',
           'maybe_release_focus', 'poll_keyboard'}
small_class = ast.ClassDef(
    name='KeyboardController', bases=[], keywords=[],
    body=[n for n in original.body if isinstance(n, ast.FunctionDef)
          and n.name in methods], decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[small_class], type_ignores=[]))


class Joy:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None)
        self.buttons = []


keys = ('1', 'q')
namespace = {'Joy': Joy, 'KEYS': keys, 'FAULTS': {'1': ('RGB', 'block'), 'q': ('LiDAR', 'drop')}}
exec(compile(module, str(SOURCE), 'exec'), namespace)
Controller = namespace['KeyboardController']


class FakeRoot:
    def __init__(self):
        self.clock = 0
        self.events = {}
        self.next_id = 0

    def after(self, delay_ms, callback, *args):
        self.next_id += 1
        self.events[self.next_id] = (self.clock + delay_ms, callback, args)
        return self.next_id

    def after_cancel(self, event_id):
        self.events.pop(event_id, None)

    def focus_displayof(self):
        return None

    def advance(self, delay_ms):
        self.clock += delay_ms
        for event_id, (deadline, callback, args) in list(self.events.items()):
            if deadline <= self.clock:
                del self.events[event_id]
                callback(*args)


class FaultHoldTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller.__new__(Controller)
        c = self.controller
        c.key_held = set()
        c.mouse_held = set()
        c.latched = set()
        c.pending_release = {}
        c.keyboard_release_debounce_ms = 180
        c.keyboard_up_counts = {}
        c.x11_key_tracking = False
        c.physical_keys = SimpleNamespace(is_down=lambda key: None)
        c.running = True
        c.root = FakeRoot()
        c.latch_mode = SimpleNamespace(get=lambda: False)
        c.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: 'stamp')),
            get_logger=lambda: SimpleNamespace(warning=lambda text: None))
        self.messages = []
        c.publisher = SimpleNamespace(publish=lambda msg: self.messages.append(msg.buttons))
        c.update_status = lambda: None

    def test_mouse_hold_persists_until_release(self):
        c = self.controller
        c.on_mouse_press('1')
        self.assertEqual(self.messages[-1], [1, 0])
        for _ in range(10):
            c.publish_once()  # 20 Hz periodic heartbeat, no state change
            self.assertEqual(self.messages[-1], [1, 0])
        c.on_mouse_release('1')
        self.assertEqual(self.messages[-1], [0, 0])

    def test_keyboard_repeat_cannot_prematurely_clear_held_key(self):
        c = self.controller
        c.on_press(SimpleNamespace(keysym='q'))
        self.assertEqual(self.messages[-1], [0, 1])
        c.on_release(SimpleNamespace(keysym='q'))  # auto-repeat fake release
        c.root.advance(90)
        c.on_press(SimpleNamespace(keysym='q'))  # next repeat cancels release
        c.root.advance(220)
        self.assertEqual(c.active(), {'q'})
        self.assertEqual(self.messages[-1], [0, 1])
        c.on_release(SimpleNamespace(keysym='q'))  # physical key release
        c.root.advance(180)
        self.assertEqual(c.active(), set())
        self.assertEqual(self.messages[-1], [0, 0])

    def test_focus_change_does_not_cancel_mouse_hold(self):
        c = self.controller
        c.on_mouse_press('1')
        c.maybe_release_focus()  # GUI may briefly lose focus while held
        self.assertEqual(c.active(), {'1'})
        self.assertEqual(self.messages[-1], [1, 0])
        c.on_mouse_release('1')
        self.assertEqual(c.active(), set())

    def test_physical_keyboard_hold_ignores_all_repeat_releases(self):
        c = self.controller
        physical_down = {'q': True}
        c.physical_keys = SimpleNamespace(is_down=lambda key: physical_down[key])
        c.x11_key_tracking = True
        c.on_press(SimpleNamespace(keysym='q'))
        self.assertEqual(c.active(), {'q'})
        for _ in range(20):  # simulate 500 ms with repeated fake releases
            c.on_release(SimpleNamespace(keysym='q'))
            c.poll_keyboard()
            self.assertEqual(c.active(), {'q'})
        physical_down['q'] = False
        c.poll_keyboard()  # one X11-up observation is not sufficient
        self.assertEqual(c.active(), {'q'})
        c.poll_keyboard()  # confirmed physical release
        self.assertEqual(c.active(), set())
        self.assertEqual(self.messages[-1], [0, 0])

    def test_latched_gui_click_toggles(self):
        c = self.controller
        c.latch_mode = SimpleNamespace(get=lambda: True)
        c.on_mouse_press('1')
        c.on_mouse_release('1')
        self.assertEqual(c.active(), {'1'})
        c.on_mouse_press('1')
        c.on_mouse_release('1')
        self.assertEqual(c.active(), set())


if __name__ == '__main__':
    unittest.main()
