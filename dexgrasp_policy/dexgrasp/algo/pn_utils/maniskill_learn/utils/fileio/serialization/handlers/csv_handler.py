import csv, io
from .base import BaseFileHandler


class CSVHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return list(csv.reader(file))

    def dump_to_fileobj(self, obj, file, **kwargs):
        csv_writer = csv.writer(file)
        csv_writer.writerows(obj)

    def dump_to_str(self, obj, **kwargs):
        output = io.StringIO()
        self.dump_to_fileobj(output, obj)
        return output.getvalue()
