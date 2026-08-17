`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/26 10:03:17
// Design Name: 
// Module Name: testbench
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


module testbench();

    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 100;
    parameter spike_window = 50;
    
    reg clock, rst,rst_cnt;
    initial begin
//        clk_n <= 1;
//        clk_p <= 0;       
        clock <= 1;
    end
    
//    always #20 clk_d <= ~clk_d;
    always #5 clock <= ~clock;
//    always #5 clk_n <= ~clk_n;
//    always #5 clk_p <= ~clk_p; 
    
    reg signed [data_width - 1 : 0] input_data;
    
    integer fp_r;
    
    initial begin
        rst <= 1;
        fp_r = $fopen ("CortexLab_Data_6groups.txt","r");
        #42
        rst <= 0;
        //#457
        #68
        repeat (3 * ch_num * 32768) begin
            $fscanf (fp_r, "%d\t", input_data);
            #80;
        end
    end 
    initial begin
        rst_cnt <= 1;
        #25
        rst_cnt <= 0;
    end
    
//    reg [data_width - 1 : 0] th, CR;
//    initial begin
//        th <= 16'd20;
//        CR <= 16'd84;
//    end
    wire [data_width * 3 - 1 : 0] ACC;
    TOP_SD_CS top(
        .clock (clock),
        .rst (rst),
        .rst_cnt (rst_cnt),
        .input_data (input_data),
        .ACC (ACC)
    );

//    wire CS_idle;  
//    reg EN_CS_idle; 
//    wire [data_width - 1 : 0] spike_counter;
//    wire wen_MEMspikelab, ren_MEMspikelab, ren_spikesegment;
//    wire signed [data_width - 1 : 0] dout_spikesegment;
    
//    reg wen_MEMspikelab1;
//    always @(posedge clk_f) begin
//        wen_MEMspikelab1 <= wen_MEMspikelab;
//    end
    
//    initial begin
//         EN_CS_idle <= 0;
//         #1540980;
//         EN_CS_idle <= 1;
//         #10
//         EN_CS_idle <= 0;
//     end

            
            
            
//    SD_STD SD(
//        .clk_d (clk_d), 
//        .clk_f (clk_f),
//        .rst (rst),
//        .input_data (input_data),
//        .th (th),
//        .CS_idle (CS_idle),
//        .spike_counter (spike_counter),
//        .wen_MEMspikelab (wen_MEMspikelab),
//        .ren_MEMspikelab (ren_MEMspikelab),
//        .ren_spikesegment (ren_spikesegment),
//        .dout_spikesegment (dout_spikesegment)
//    );
    
//    CS_MDC CS(
//        .clk_d (clk_d), 
//        .clk_f (clk_f),
//        .rst (rst),
//        .EN_CS_idle (EN_CS_idle),
//        .CR (CR),
//        .ren_spikesegment (ren_spikesegment),
//        .dout_spikesegment (dout_spikesegment),
//        .CS_idle (CS_idle)
//    );
endmodule
