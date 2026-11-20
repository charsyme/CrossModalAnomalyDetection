#!/usr/bin/env python3
"""Continuous press-and-hold fault controls over /faults/keys (Joy).

GUI mouse holds end only on the corresponding mouse release. On Linux/X11,
keyboard holds follow actual key states, ignoring synthetic key-repeat
releases. Other desktops use a Tk event-based fallback.
"""
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from .fault_effects import FAULTS, KEYS, GROUPS
from .x11_key_state import X11KeyState


class KeyboardController:
    def __init__(self):
        rclpy.init()
        self.node = Node('fault_controller')
        topic = self.node.declare_parameter('keys_topic','/faults/keys').value
        self.publisher = self.node.create_publisher(Joy,topic,10)
        # Fallback only for platforms on which XQueryKeymap cannot be used.
        self.keyboard_release_debounce_ms = max(40, int(self.node.declare_parameter(
            "keyboard_release_debounce_ms", 240).value))
        self.physical_keys = X11KeyState()
        self.x11_key_tracking = self.physical_keys.available
        self.keyboard_up_counts = {}
        self.key_held = set()
        self.mouse_held = set()
        self.latched = set()
        self.pending_release = {}
        self.latch_mode = None
        self.running = True
        self.buttons = {}

        self.root = tk.Tk()
        self.root.title('TurtleBot4 RGB / LiDAR fault simulator — 16 selected faults')
        # Narrow single-column layout: only the 16 selected faults appear.
        self.root.geometry('560x760')
        self.root.minsize(420,520)
        self.root.configure(padx=12,pady=10)

        tk.Label(self.root,text=f'Multi-sensor fault injection controller ({len(FAULTS)} faults)',
                 font=('Sans',16,'bold')).pack(anchor='w')
        tk.Label(self.root,
                 text='HOLD a keyboard key or GUI button to keep its fault ACTIVE for as long as it is pressed. '
                      'Release it to restore the original sensor stream. Original sensor topics are NEVER modified.',
                 wraplength=515,justify='left').pack(anchor='w',pady=(5,6))
        self.latch_mode = tk.BooleanVar(value=False)
        tools = tk.Frame(self.root)
        tools.pack(fill='x',pady=(4,7))
        tk.Checkbutton(tools,text='Toggle GUI buttons instead of holding',
                       variable=self.latch_mode).pack(side='left')
        tk.Button(tools,text='CLEAR ALL  (Esc / Space)',command=self.clear_all,
                  bg='#f7c6c6',font=('Sans',10,'bold')).pack(side='right')

        # One continuous, scrollable column with compact group headings.
        # KEYS order and Joy button indices match fault_effects.KEYS.
        panel = tk.Frame(self.root)
        panel.pack(fill='both', expand=True)
        canvas = tk.Canvas(panel, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(panel, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        fault_list = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=fault_list, anchor='nw')
        fault_list.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window_id, width=e.width))

        # Linux/X11 and Windows scroll-wheel handling.
        def scroll_wheel(event):
            if getattr(event, 'num', None) == 4:
                delta = -1
            elif getattr(event, 'num', None) == 5:
                delta = 1
            else:
                delta = -int(event.delta / 120) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, 'units')
        canvas.bind_all('<MouseWheel>', scroll_wheel)
        canvas.bind_all('<Button-4>', scroll_wheel)
        canvas.bind_all('<Button-5>', scroll_wheel)

        for group, keys in GROUPS.items():
            tk.Label(
                fault_list, text=group, font=('Sans', 10, 'bold'),
                anchor='w', bg='#dce7f1', padx=10, pady=5,
            ).pack(fill='x', padx=4, pady=(8, 3))
            for key in keys:
                name = FAULTS[key][1]
                button = tk.Button(
                    fault_list, text=f'[{key.upper()}]  {name}',
                    font=('Sans', 10), wraplength=400,
                    justify='left', anchor='w', padx=10, pady=5,
                    relief='raised', bg='#ededed', activebackground='#d2e9ff',
                )
                button.pack(fill='x', padx=7, pady=2)
                button.bind('<ButtonPress-1>', lambda event, k=key: self.on_mouse_press(k))
                button.bind('<ButtonRelease-1>', lambda event, k=key: self.on_mouse_release(k))
                self.buttons[key] = button
        self.status=tk.Label(self.root,text='Active: none',font=('Sans',11,'bold'),
                             wraplength=500,justify='left',anchor='w')
        self.status.pack(fill='x',pady=(7,3))
        tk.Label(self.root,text='Safety: route fault outputs ONLY to experimental nodes, never robot control/navigation. '
                  'Keyboard holds are cleared if the window loses focus. GUI faults remain enabled until mouse release; toggle faults remain until cleared.',
                  fg='#6b3a33',wraplength=515,justify='left').pack(anchor='w')
        self.root.bind_all('<KeyPress>',self.on_press)
        self.root.bind_all('<KeyRelease>',self.on_release)
        self.root.bind('<FocusOut>',self.on_focus_out)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(50,self.publish_state)  # 20 Hz heartbeat, including unchanged held states
        self.root.after(25,self.poll_keyboard)  # physical key state / release confirmation
        self.root.after(120,self.root.focus_force)
        self.node.get_logger().info(
            "Keyboard hold tracking: " + ("X11 physical key state" if self.x11_key_tracking
                                       else "Tk events with release debounce"))

    def active(self):
        return (self.key_held | self.mouse_held | self.latched) & set(FAULTS)

    def update_status(self):
        a=self.active()
        self.status.configure(text='Active: '+(', '.join(f'{k.upper()} — {FAULTS[k][1]}'
                                                for k in KEYS if k in a) or 'none'))
        for k,button in self.buttons.items():
            enabled=k in a
            button.configure(bg='#ace0b2' if enabled else '#ededed',
                             relief='sunken' if enabled else 'raised')

    def on_press(self, event):
        key = event.keysym.lower()
        if key in ('escape', 'space'):
            self.clear_all()
            return
        if key not in FAULTS:
            return
        if key in self.pending_release:
            self.root.after_cancel(self.pending_release.pop(key))
        self.keyboard_up_counts.pop(key, None)
        # X11 physical key state should already be down when a real Tk KeyPress
        # is dispatched. If an XWayland backend reports otherwise, gracefully
        # fall back to Tk events rather than suppressing legitimate key holds.
        if self.x11_key_tracking and self.physical_keys.is_down(key) is False:
            self.x11_key_tracking = False
            self.node.get_logger().warning(
                'X11 physical key-state reporting unavailable; using Tk fallback')
        if key not in self.key_held:
            self.key_held.add(key)
            self.update_status()
            self.publish_once()

    def on_release(self, event):
        key = event.keysym.lower()
        if key not in FAULTS:
            return
        if self.x11_key_tracking:
            # Ignore KeyRelease entirely: X11 emits fake releases during
            # auto-repeat. poll_keyboard() will detect actual physical release.
            return
        if key in self.pending_release:
            self.root.after_cancel(self.pending_release.pop(key))
        self.pending_release[key] = self.root.after(
            self.keyboard_release_debounce_ms, self.finish_release, key)

    def poll_keyboard(self):
        """Verify physical held keys without mistaking auto-repeat for release."""
        if not self.running:
            return
        if self.x11_key_tracking:
            changed = False
            for key in tuple(self.key_held):
                pressed = self.physical_keys.is_down(key)
                if pressed is None:
                    # An unusual keyboard mapping: use fallback for the whole
                    # controller rather than leaving any key stuck indefinitely.
                    self.x11_key_tracking = False
                    break
                if pressed:
                    self.keyboard_up_counts.pop(key, None)
                else:
                    count = self.keyboard_up_counts.get(key, 0) + 1
                    if count >= 2:  # confirm across two 25-ms polls
                        self.keyboard_up_counts.pop(key, None)
                        self.key_held.discard(key)
                        changed = True
                    else:
                        self.keyboard_up_counts[key] = count
            if changed:
                self.update_status()
                self.publish_once()
        self.root.after(25, self.poll_keyboard)

    def finish_release(self, key):
        self.pending_release.pop(key, None)
        if key in self.key_held:
            self.key_held.discard(key)
            self.update_status()
            self.publish_once()

    def on_mouse_press(self,key):
        if self.latch_mode.get():
            if key in self.latched:self.latched.remove(key)
            else:self.latched.add(key)
        else:
            self.mouse_held.add(key)
        self.update_status()
        self.publish_once()

    def on_mouse_release(self,key):
        self.mouse_held.discard(key)
        self.update_status()
        self.publish_once()

    def clear_all(self):
        for v in self.pending_release.values():
            self.root.after_cancel(v)
        self.pending_release.clear()
        self.key_held.clear()
        self.mouse_held.clear()
        self.keyboard_up_counts.clear()
        self.latched.clear()
        self.update_status()
        self.publish_once()

    def on_focus_out(self,_event):
        # Ignore moving focus within the same application. Never cancel a
        # mouse hold on FocusOut: it belongs to the ButtonRelease event.
        self.root.after(70,self.maybe_release_focus)

    def maybe_release_focus(self):
        if self.root.focus_displayof() is not None:
            return
        for v in self.pending_release.values():
            self.root.after_cancel(v)
        self.pending_release.clear()
        # Window focus may change during an active GUI mouse drag.  FocusOut
        # must *never* cancel mouse faults; ButtonRelease-1 owns their lifetime.
        self.key_held.clear()
        self.keyboard_up_counts.clear()
        self.update_status()
        self.publish_once()

    def publish_once(self):
        msg=Joy()
        msg.header.stamp=self.node.get_clock().now().to_msg()
        a=self.active()
        msg.buttons=[int(k in a and k in FAULTS) for k in KEYS]
        self.publisher.publish(msg)

    def publish_state(self):
        if not self.running:
            return
        self.publish_once()
        self.root.after(50,self.publish_state)

    def close(self):
        self.clear_all()
        self.running=False
        self.root.destroy()
        self.physical_keys.close()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def run(self):
        self.root.mainloop()


def main():
    KeyboardController().run()


if __name__ == '__main__':
    main()
