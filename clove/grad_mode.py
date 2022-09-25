import contextlib


class _GradController:
    instance = None

    def __init__(self):
        self.grad_enabled = True

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


def _set_grad_mode(mode: bool):
    controller = _GradController.get_instance()
    controller.grad_enabled = mode


def is_grad_enabled():
    controller = _GradController.get_instance()
    return controller.grad_enabled


@contextlib.contextmanager
def set_grad_enabled(mode):
    try:
        prev_context_mode = is_grad_enabled()
        _set_grad_mode(mode)
        yield
    finally:
        _set_grad_mode(prev_context_mode)


@contextlib.contextmanager
def no_grad():
    try:
        prev_context_mode = is_grad_enabled()
        _set_grad_mode(False)
        yield
    finally:
        _set_grad_mode(prev_context_mode)
