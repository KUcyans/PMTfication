from EventFilter import EventFilter

class CompositeEventFilter(EventFilter):
    def __init__(self, source_subdir, output_subdir, subdir_no, part_no, valid_event_nos, filter_keyword: str):
        self.filter_keyword = filter_keyword
        super().__init__(source_subdir, output_subdir, subdir_no, part_no)
        self.valid_event_nos = valid_event_nos

        self.logger.info(f"Initialized ({self.filter_keyword})")

    def _set_valid_event_nos(self):
        pass
