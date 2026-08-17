`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/09 10:26:03
// Design Name: 
// Module Name: BRAM
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


module BRAM(
    clk,
    rst,
    wen_MEM200, 
    ren_MEM200,
    waddr_MEM200, 
    raddr_MEM200,
    input_data,
    dout_MEM200
    );
    
    parameter data_width = 16;
    parameter ch_num = 384;
    parameter L_MEM = 200;
    
    input clk;
    input rst;
    
    input wen_MEM200, ren_MEM200;
    input [$clog2(L_MEM * ch_num) - 1 : 0] waddr_MEM200, raddr_MEM200;
    input signed [data_width - 1 : 0] input_data;
    output wire signed [data_width - 1 : 0] dout_MEM200;
    (* ram_style = "block" *) reg signed [data_width - 1 : 0] MEM_200 [L_MEM * ch_num - 1 : 0];
    
    always @(posedge clk) begin
        if (wen_MEM200) begin
            MEM_200[waddr_MEM200] <= input_data;
        end
    end
    
    assign dout_MEM200 = ren_MEM200 ? MEM_200[raddr_MEM200] : 0;
endmodule
