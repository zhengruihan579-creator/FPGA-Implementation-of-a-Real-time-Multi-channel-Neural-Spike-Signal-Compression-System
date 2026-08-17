`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/03 08:48:20
// Design Name: 
// Module Name: MEM_3data
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


module MEM_3data(
    clk_d, clk_f,
    rst,
    input_data,
    wen_MEM3data, 
    ren_MEM3data,
    waddr_MEM3data, 
    radda_MEM3data,
    dout_MEM3data
    );
    
    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 100;
    parameter th = 20;
    parameter spike_window = 50;
    
    parameter N_possibel_sp = 350000;
    
    input clk_d, clk_f;
    input rst;
    input signed [data_width - 1 : 0] input_data;
    
    input wen_MEM3data, ren_MEM3data;
    input [$clog2(ch_num) - 1 : 0] waddr_MEM3data, radda_MEM3data;
    output wire [3 * data_width - 1 : 0] dout_MEM3data;
    
    (* ram_style = "block" *) reg [3 * data_width - 1 : 0] MEM_3data [ch_num - 1 : 0];
    
    reg [2 : 0] counter = 0;
    always @(posedge clk_d) begin
//        if (rst) begin
//            counter <= 0;
//        end
        if (waddr_MEM3data == ch_num - 1) begin
            if (counter < 2) begin
                counter <= counter + 1;
            end
            else begin
                counter <= 0;
            end
        end
        else begin
            counter <= counter;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEM3data) begin
            MEM_3data[waddr_MEM3data][data_width * counter +: data_width] <= input_data;
        end
    end
    
    assign dout_MEM3data = (ren_MEM3data) ? MEM_3data[radda_MEM3data] : 0;
    
endmodule
