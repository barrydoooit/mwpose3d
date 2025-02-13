# Quick Start

Run the following command:

```bash
python apps/main.py --type cvis
```

# CLI Commands (Bottom Left)
| **Command Name** | **Description**                                                                                         | **Usage**                    |
|------------------|---------------------------------------------------------------------------------------------------------|------------------------------|
| `buf`            | Sets the maximum buffer size.                                                                          | `buf <buffer_size>`          |
| `out`            | Sets the output directory for saving data.                                                             | `out <output_dir>`           |
| `mid`            | Sets the mid break length (in seconds) used for buffering.                                             | `mid <mid_break_len>`        |
| `mod`            | Switches the operating mode (only when the loop is stopped). Available modes: `collect`, `visualize`.  | `mod <mode>`                 |
| `start`          | Starts the data collection loop and resets the visuals.                                               | `start`                      |
| `stop`           | Stops the data collection loop and clears the buffer.                                                 | `stop`                       |
