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


module TOP_module(
    clk_p,
    clk_n,
    rst,
//    I_spi_cs, 
//    I_spi_clk, 
//    I_spi_mosi,
    ACC
    );
    
    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 100;
    parameter spike_window = 50;
    
    input clk_p, clk_n;
    input rst;
//    input I_spi_cs, I_spi_clk;
//    input I_spi_mosi;
    output ACC;
    
    wire clock;

    clk_wiz_0 C_W1
   (
    // Clock out ports
    .clk_out1(clock),     // output clk_out1
    // Status and control signals
   
   // Clock in ports
    .clk_in1_p(clk_p),    // input clk_in1_p
    .clk_in1_n(clk_n));    // input clk_in1_n
   
    reg rst_cnt = 1;
    reg [$clog2(100) - 1 : 0] rst_cnt_counter = 0;
    always @(posedge clock) begin
        if (rst_cnt_counter < 100) begin
            rst_cnt_counter <= rst_cnt_counter + 1;
        end
        else begin
            rst_cnt_counter <= rst_cnt_counter;
        end
    end
    
    always @(posedge clock) begin
        if (rst_cnt_counter >= 2) begin
            rst_cnt <= 0;
        end
        else begin 
            rst_cnt <= rst_cnt;
        end
    end
    

//    wire O_spi_rvalid;
//    wire signed[data_width-1:0] O_spi_rdata;
//    wire signed[data_width-1:0] input_data;
//    assign input_data = O_spi_rvalid ? O_spi_rdata : input_data;
    
//     spi_module u_spi_slave
//     (
//        .sys_clk        ( clock           ),
//        .rst            ( rst           ),
//        .spi_cs         ( I_spi_cs      ),
//        .spi_clk        ( I_spi_clk     ),
//        .spi_mosi       ( I_spi_mosi    ),
//        .rx_data_valid  ( O_spi_rvalid  ),
//        .rx_data        ( O_spi_rdata   )
//    );
    
    
    wire [data_width * 3 - 1 : 0] ACC;
    
    TOP_SD_CS top(
        .clock (clock),
        .rst (rst),
        .rst_cnt (rst_cnt),
//        .input_data (input_data),
        .ACC (ACC)
        );
endmodule
