"""Shared Modal App and Volume — import from here to keep all functions on one App."""
import modal

app = modal.App("neural-fm")
data_volume = modal.Volume.from_name("project", create_if_missing=False)
