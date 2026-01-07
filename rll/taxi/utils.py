def episilon_decay_builder(decay_end: float, start: float = 0.5, end: float = 0.01):
    def episilon_decay(ep: float):
        if ep >= decay_end:
            return end
        return start + (end - start) * (ep / decay_end)

    return episilon_decay
