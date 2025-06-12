module TC0360PRI #(parameter SS_IDX=-1) (
    input clk,
    input ce_pixel,
    input reset,

    // CPU Interface
    input [7:0] cpu_din,
    output reg [7:0] cpu_dout,

    input        cs,
    input [3:0]  cpu_addr,
    input        cpu_rw,
    input        cpu_ds_n,

    input [13:0] color_in0 /* verilator public_flat */,
    input [13:0] color_in1 /* verilator public_flat */,
    input [5:0]  color_in2 /* verilator public_flat */,
    output [13:0] color_out /* verilator public_flat */,

    ssbus_if.slave ssbus
);

reg [7:0] ctrl[16];

always_ff @(posedge clk) begin
    if (reset) begin
        int i;
        for (int i = 0; i < 16; i = i + 1) begin
            ctrl[i] <= 0;
        end
    end else begin
        if (cs) begin
            if (cpu_rw) begin
                cpu_dout <= ctrl[cpu_addr[3:0]];
            end else begin
                if (~cpu_ds_n) ctrl[cpu_addr[3:0]]  <= cpu_din;
            end
        end

        ssbus.setup(SS_IDX, 16, 0);
        if (ssbus.access(SS_IDX)) begin
            if (ssbus.write) begin
                ctrl[ssbus.addr[3:0]] <= ssbus.data[7:0];
                ssbus.write_ack(SS_IDX);
            end else if (ssbus.read) begin
                ssbus.read_response(SS_IDX, { 56'b0, ctrl[ssbus.addr[3:0]] });
            end
        end
    end
end


wire [1:0] sel0 = color_in0[13:12];
wire [1:0] sel1 = color_in1[13:12];
wire [1:0] sel2 = ctrl[1][7:6];

wire [15:0] prio_vals0 = { ctrl[5], ctrl[4] };
wire [15:0] prio_vals1 = { ctrl[7], ctrl[6] };
wire [15:0] prio_vals2 = { ctrl[9], ctrl[8] };

wire [4:0] prio0 = |color_in0[3:0] ? { 1'b0, prio_vals0[ 4 * sel0 +: 4 ] } : 5'b0;
wire [4:0] prio1 = |color_in1[3:0] ? { 1'b0, prio_vals1[ 4 * sel1 +: 4 ] } : 5'b0;
wire [4:0] prio2 = |color_in2[3:0] ? { 1'b0, prio_vals2[ 4 * sel2 +: 4 ] } : 5'b0;

wire [11:0] color0 = color_in0[11:0];
wire [11:0] color1 = color_in1[11:0];
wire [11:0] color2 = { ctrl[1][5:0], color_in2 };

wire blend = ctrl[0][7];
wire bm1 = ctrl[0][7] & ctrl[0][6];
wire bm2 = ctrl[0][7] & ~ctrl[0][6];

reg [11:0] color_final;

assign color_out = { 2'b0, color_final };

always_ff @(posedge clk) begin
    if (ce_pixel) begin
        color_final <= color0;
        if (blend && (prio1 == (prio0 - 5'd1))) begin
            if (bm1) begin
                color_final <= { color1[11:4], color0[3:0] };
            end else begin
                color_final <= { color0[11:5], 1'b0, color0[3:0] };
            end
        end else if (blend && (prio1 == (prio0 + 5'd1))) begin
            if (bm1) begin
                color_final <= { color0[11:4], color1[3:0] };
            end else begin
                color_final <= { color1[11:5], 1'b0, color1[3:0] };
            end
        end else if (blend && (prio1 == (prio2 - 5'd1))) begin
            if (bm1) begin
                color_final <= { color1[11:4], color2[3:0] };
            end else begin
                color_final <= { color2[11:5], 1'b0, color2[3:0] };
            end
        end else if (blend && (prio1 == (prio2 + 5'd1))) begin
            if (bm1) begin
                color_final <= { color2[11:4], color1[3:0] };
            end else begin
                color_final <= { color1[11:5], 1'b0, color1[3:0] };
            end
        end else if (prio1 > prio0) begin
            if (prio2 > prio1) begin
                color_final <= color2;
            end else begin
                color_final <= color1;
            end
        end else if (prio2 > prio0) begin
            color_final <= color2;
        end
    end
end


endmodule
