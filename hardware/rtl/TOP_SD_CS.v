`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/27 09:37:41
// Design Name: 
// Module Name: TOP_SD_CS
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module TOP_SD_CS(
    clock,
    rst,
    rst_cnt,
    input_data,
    ACC
    );
    
    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 100;
    parameter spike_window = 50;
    
    input clock;
    input rst, rst_cnt;
    input signed [data_width - 1 : 0] input_data;
    output ACC;
    
    reg clk_d, clk_f;
    
    reg [$clog2(8) - 1 : 0] cnt_clkd;
    always @(posedge clock) begin
        if (rst_cnt) begin
            cnt_clkd <= 0;
        end
        else if (cnt_clkd < 8 - 1) begin
            cnt_clkd <= cnt_clkd + 1;
        end
        else begin
            cnt_clkd <= 0;
        end
    end
    
    always @(posedge clock) begin
        if (rst_cnt) begin
            clk_d <= 0;
        end
        else if (cnt_clkd == 0) begin
            clk_d <= ~clk_d;
        end
        else if (cnt_clkd == 4) begin
            clk_d <= ~clk_d;
        end
        else begin
            clk_d <= clk_d;
        end
    end
    
    reg [$clog2(2) - 1 : 0] cnt_clkf;
    always @(posedge clock) begin
        if (rst_cnt) begin
            cnt_clkf <= 0;
        end
        else if (cnt_clkf <= 2 - 1) begin
            cnt_clkf <= cnt_clkf + 1;
        end
        else begin
            cnt_clkf <= 0;
        end
    end
    
    always @(posedge clock) begin
        if (rst_cnt) begin
            clk_f <= 0;
        end
        else if (cnt_clkf == 0) begin
            clk_f <= ~clk_f;
        end
        else if (cnt_clkf == 1) begin
            clk_f <= ~clk_f;
        end
        else begin 
            clk_f <= clk_f;
        end
    end
    
    wire CS_idle;  
    reg EN_CS, EN_CS_idle; 
    wire [data_width - 1 : 0] spike_counter;
    wire wen_MEMspikelab, ren_MEMspikelab, ren_spikesegment;
    wire signed [data_width - 1 : 0] dout_spikesegment;

    reg [$clog2(312050) - 1 : 0] EN_CS_idle_counter;
    
    always @(posedge clock) begin
        if (rst) begin
            EN_CS_idle_counter <= 0;
        end
        else if (EN_CS_idle_counter < 308183) begin
            EN_CS_idle_counter <= EN_CS_idle_counter + 1;
        end
        else begin
            EN_CS_idle_counter <= EN_CS_idle_counter;
        end
    end
    
    always @(posedge clock) begin
        if (rst) begin
            EN_CS <= 0;
        end
        else if ((EN_CS_idle_counter == 308181) || (EN_CS_idle_counter == 308182)) begin
            EN_CS <= 1;
        end
        else begin
            EN_CS <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            EN_CS_idle <= 0;
        end
        else begin
            EN_CS_idle <= EN_CS;
        end
    end
//    reg [4 : 0] ena_addr;
//    always @(posedge clock) begin
//        if (rst) begin
//            ena_addr <= 0;
//        end
//        else if (ena_addr < 6) begin
//            ena_addr <= ena_addr + 1;
//        end
//        else begin
//            ena_addr <= ena_addr;
//        end
//    end
    reg ena;
    always @(posedge clock) begin
        if (rst) begin
            ena <= 0;
        end
//        else if (ena_addr >= 5) begin
//            ena <= 1;
//        end
        else begin
            ena <= 1;
        end
    end
    
    reg [19 : 0] addra;
    always @(posedge clk_d) begin
        if (rst) begin
            addra <= 0;
        end
        else if (ena) begin
            addra <= addra + 1;
        end
        else begin
            addra <= 0;
        end
    end
    
//    wire signed [data_width - 1 : 0] input_data;
//    blk_mem_gen_1 your_instance_name (
//        .clka(clk_d),    // input wire clka
//        .ena(ena),      // input wire ena
//        .addra(addra),  // input wire [19 : 0] addra
//        .douta(input_data)  // output wire [15 : 0] douta
//);

    SD_STD SD(
        .clk_d (clk_d), 
        .clk_f (clk_f),
        .rst (rst),
        .input_data (input_data),
        .CS_idle (CS_idle),
        .spike_counter (spike_counter),
        .wen_MEMspikelab (wen_MEMspikelab),
        .ren_MEMspikelab (ren_MEMspikelab),
        .ren_spikesegment (ren_spikesegment),
        .dout_spikesegment (dout_spikesegment)
    );
    
    wire [data_width * 3 - 1 : 0] ACC;
    CS_MDC CS(
        .clk_d (clk_d), 
        .clk_f (clk_f),
        .rst (rst),
        .EN_CS_idle (EN_CS_idle),
        .ren_spikesegment (ren_spikesegment),
        .dout_spikesegment (dout_spikesegment),
        .CS_idle (CS_idle),
        .ACC (ACC)
    );
endmodule
