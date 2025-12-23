try:
    from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QVBoxLayout
    from PySide6.QtCore import Signal
except ImportError:
    pass



class InputPopupDialog(QDialog):
    # Emitted when the dialog is accepted, with a dict of non-empty inputs
    submitted = Signal(dict)

    _labels = []
    _previous_values = {}

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.setModal(True)
        self._fields = {}
        self.set_labels(labels)
        self._build_ui()

    def _build_ui(self):
        # Construct form and buttons
        layout = QVBoxLayout(self)
        self._form = QFormLayout()
        layout.addLayout(self._form)

        # OK button only
        buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=self)
        buttons.accepted.connect(self._on_accept)
        layout.addWidget(buttons)

        # Initialize fields
        self._create_fields()

    def _create_fields(self):
        # Clear any existing rows
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._fields.clear()

        for key in self._labels:
            field = QLineEdit(self)
            if key in self._previous_values:
                field.setText(self._previous_values[key])
            self._form.addRow(f"{key}:", field)
            self._fields[key] = field

    def refresh_fields(self):
        # Call to re-create fields from stored previous values
        self._create_fields()

    @classmethod
    def get_labels(cls):
        return cls._labels

    @classmethod
    def set_labels(cls, labels: list[str]):
        cls._labels = labels.copy()

    @classmethod
    def set_previous_values(cls, values: dict):
        # Store values for next opening
        cls._previous_values = values.copy()

    def _on_accept(self):
        # Collect non-empty inputs and store them statically
        result = {key: fld.text() for key, fld in self._fields.items() if fld.text().strip()}
        InputPopupDialog.set_previous_values(result)
        self.submitted.emit(result)
        self.accept()