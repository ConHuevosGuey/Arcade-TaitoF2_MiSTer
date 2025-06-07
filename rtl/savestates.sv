interface ssbus_if();
    logic [63:0] data;
    logic [31:0] addr;
    logic [7:0] select;
    logic write;
    logic read;
    logic query;
    logic [63:0] data_out;
    logic ack;

    function logic access(int idx);
        return (select == idx[7:0]) & ~query & (read | write);
    endfunction

    task setup(int idx, input [31:0] count, int width);
        ack <= 0;
        if (select == idx[7:0]) begin
            if (query) begin
                data_out <= { idx[7:0], 22'b0, width[1:0], count };
                ack <= 1;
            end
        end
    endtask

    task read_response(int idx, input [63:0] dout);
        if (select == idx[7:0]) begin
            data_out <= dout;
            ack <= 1;
        end
    endtask

    task write_ack(int idx);
        if (select == idx[7:0]) begin
            ack <= 1;
        end
    endtask

    modport master(
        output data, addr, select, write, read, query,
        input data_out, ack
    );

    modport slave(
        input data, addr, select, write, read, query,
        output data_out, ack,
        import access,
        import setup,
        import read_response,
        import write_ack
    );
endinterface

module ssbus_mux #(parameter COUNT = 4)(
    input clk,
    ssbus_if.master masters[COUNT],
    ssbus_if.slave slave
);

logic[63:0] data_out[COUNT];
logic ack[COUNT];

genvar gi;
generate
for (gi = 0; gi < COUNT; gi = gi + 1) begin: gen_loop
    always_comb begin
        ack[gi] = masters[gi].ack;
        data_out[gi] = masters[gi].data_out;

        masters[gi].data = slave.data;
        masters[gi].addr = slave.addr;
        masters[gi].select = slave.select;
        masters[gi].write = slave.write;
        masters[gi].read = slave.read;
        masters[gi].query = slave.query;
    end
end
endgenerate

int i;
always_ff @(posedge clk) begin
    slave.data_out <= 64'd0;
    slave.ack <= 0;

    for (i = 0; i < COUNT; i = i + 1) begin
        if (ack[i]) begin
            slave.ack <= 1;
            slave.data_out <= data_out[i];
        end
    end
end

endmodule

module auto_save_adaptor #(parameter N_BITS=16, SS_IDX=-1)(
    input clk,

    ssbus_if.slave ssbus,

    input [N_BITS-1:0] bits_in,
    output [N_BITS-1:0] bits_out,
    output bits_wr
);

reg [(N_WORDS * 16) - 1:0] storage;
reg [(N_WORDS * 16) - 1:0] storage1;

assign bits_out = storage[N_BITS-1:0];

localparam N_WORDS = (N_BITS + 15) / 16;

always @(posedge clk) begin
    storage1[N_BITS-1:0] <= bits_in;
    bits_wr <= 0;
    ssbus.setup(SS_IDX, N_WORDS + 1, 1); // FIXME do 64-bit writes once verified

    if (ssbus.access(SS_IDX)) begin
        if (ssbus.write) begin
            if (ssbus.addr == N_WORDS ) begin
                bits_wr <= 1;
            end else begin
                storage[ ssbus.addr * 16 +: 16 ] <= ssbus.data[15:0];
            end
            ssbus.write_ack(SS_IDX);
        end else if (ssbus.read) begin
            ssbus.read_response(SS_IDX, {48'd0, storage1[ ssbus.addr * 16 +: 16 ] } );
        end
    end

end

endmodule

module auto_save_adaptor2 #(parameter DEVICE_WIDTH=8, STATE_WIDTH=16, SS_IDX=-1)(
    input clk,

    ssbus_if.slave ssbus,

    output reg rd,
    output reg wr,
    output reg [DEVICE_WIDTH-1:0] device_idx,
    output reg [STATE_WIDTH-1:0] state_idx,
    output reg [31:0] wr_data,
    input      [31:0] rd_data,
    input      ack
);

typedef enum { ST_INIT, ST_WAIT_READ, ST_CHECK_READ, ST_READY } state_t;
state_t state = ST_INIT;

reg [15:0] count;

always @(posedge clk) begin
    case(state)
        ST_INIT: begin
            count <= 0;
            wr <= 0;
            rd <= 1;
            device_idx <= 0;
            state_idx <= 0;
            state <= ST_WAIT_READ;
        end

        ST_WAIT_READ: begin
            state <= ST_CHECK_READ;
        end

        ST_CHECK_READ: begin
            if (ack) begin
                count <= count + 1;
                state_idx <= state_idx + 1;
                state <= ST_WAIT_READ;
            end else begin
                state_idx <= 0;
                if (&device_idx) begin
                    state <= ST_READY;
                end else begin
                    device_idx <= device_idx + 1;
                    state <= ST_WAIT_READ;
                end
            end

        end

        ST_READY: begin
            rd <= 0;
            wr <= 0;
        end

    endcase
end

endmodule


module save_state_data(
    input clk,
    input reset,

    ddr_if.to_host ddr,

    input             read_start,
    input             write_start,
    input  [1:0]      index,
    output            busy,

    ssbus_if.master   ssbus
);

// Instantiate the memory_stream module
memory_stream memory_stream (
    .clk(clk),
    .reset(reset),

    // Memory interface
    .ddr,

    .read_req(ssbus.read),
    .read_data(ssbus.data_out),
    .data_ack(ssbus.ack),

    .write_req(ssbus.write),
    .write_data(ssbus.data),

    .start_addr(SS_DDR_BASE + (index * 32'h00200000)),
    .length(32'h00200000),
    .read_start(read_start),
    .write_start(write_start),
    .busy(busy),

    .chunk_select(ssbus.select),
    .chunk_address(ssbus.addr),
    .query_req(ssbus.query)
);


endmodule

