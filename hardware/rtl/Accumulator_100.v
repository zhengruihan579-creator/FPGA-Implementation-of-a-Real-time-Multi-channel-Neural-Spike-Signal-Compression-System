`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/01/15 12:32:05
// Design Name: 
// Module Name: Accmulator_160
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


module Accumulator_100(
    clock,
    rst,
    en,
    data_in,
    acc_out
    );
    
    parameter data_num = 10;
    parameter data_width = 32;
    
    input clock, rst, en;
    input [data_num*data_width-1:0] data_in;
    
    output acc_out;
    
    wire [data_width-1:0]in0, in1, in2, in3, in4, in5, in6, in7, in8, in9;
    
    
    assign in0 = data_in [data_width*1-1:data_width*0];
    assign in1 = data_in [data_width*2-1:data_width*1];
    assign in2 = data_in [data_width*3-1:data_width*2];
    assign in3 = data_in [data_width*4-1:data_width*3];
    assign in4 = data_in [data_width*5-1:data_width*4];
    assign in5 = data_in [data_width*6-1:data_width*5];
    assign in6 = data_in [data_width*7-1:data_width*6];
    assign in7 = data_in [data_width*8-1:data_width*7];
    assign in8 = data_in [data_width*9-1:data_width*8];
    assign in9 = data_in [data_width*10-1:data_width*9];
    
    reg [47:0] acc_out;
    always @(posedge clock) begin
        if(rst) begin
            acc_out <= 0;
        end
        else if(en) begin
            acc_out <= in0 + in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8 + in9;
        end
    end
    
endmodule
