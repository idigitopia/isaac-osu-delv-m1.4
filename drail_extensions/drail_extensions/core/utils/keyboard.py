"""Keyboard interactive control."""

import weakref

import carb
import omni


class Keyboard:
    """Keyboard control base class for interactive control."""

    def __init__(self):
        # Acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # Current key pressed
        self._current_key = None

    """Public method."""

    def get_input(self):
        """Get the current key pressed."""
        return self._current_key

    """Internal methods."""

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._current_key = event.input.name
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._current_key = None


class RawKeyboard:
    """Keyboard control using pynput for direct keyboard access."""

    def __init__(self):
        from pynput import keyboard

        self._keyboard = keyboard
        self._current_key = None

        # Initialize keyboard listener with weakref to avoid circular references
        self._listener = keyboard.Listener(
            on_press=lambda key, obj=weakref.proxy(self): obj._on_press(key),
            on_release=lambda key, obj=weakref.proxy(self): obj._on_release(key),
        )
        self._listener.start()

    """Public method."""

    def get_input(self):
        """Get the current key pressed."""
        return self._current_key

    """Internal methods."""

    def __del__(self):
        """Stop the keyboard listener."""
        if hasattr(self, "_listener"):
            self._listener.stop()
            self._listener = None

    def _on_press(self, key):
        """Handle key press events."""
        try:  # noqa: SIM105
            if isinstance(key, self._keyboard.Key):
                self._current_key = key.name
            elif hasattr(key, "char"):
                self._current_key = key.char.lower()
        except AttributeError:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        self._current_key = None
