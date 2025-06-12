state_module.py current generates verilog code that saves and restores state as a single massive vector. This is inefficent because it uses a large amount of FPGA resources. We need to change it so data can be read and written in smaller chunks. Below I describe the main parts of the design.

# Ports
```
input auto_ss_rd
input auto_ss_wr
input [DATA_WIDTH-1:0] auto_ss_data_in
input [DEVICE_WIDTH-1:0] auto_ss_device_idx
input [STATE_WIDTH-1:0] auto_ss_state_idx

input [DEVICE_WIDTH-1:0] auto_ss_base_device_idx

output [DATA_WIDTH-1:0] auto_ss_data_out
output auto_ss_ack
```

# Usage
`auto_ss_device_idx` and `auto_ss_state_idx` form a two part address. The device index selects a module instance and the state index selects the piece of state within the module.

`auto_ss_base_device_idx` is a constant that specifies the device index for this instance. `Module.ancestor_count()` should be used to calculate this.

# Config
`DATA_WIDTH`, `STATE_WIDTH` and `DEVICE_WIDTH` should be configurable via the `Config` object in state_module.py

# Packing
If we know the exact size of state values, then as many of them as possible should be packed into a single data chunk. Unpacked arrays should use a data chunk per array index, don't try to pack them.


