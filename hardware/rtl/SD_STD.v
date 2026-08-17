`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/24 16:37:26
// Design Name: 
// Module Name: SD_STD
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


module SD_STD(
    clk_d, 
    clk_f,
    rst,
    input_data,
    CS_idle,
    spike_counter,
    wen_MEMspikelab,
    ren_MEMspikelab,
    ren_spikesegment,
    dout_spikesegment
    );

    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 500;
    parameter th = 20;
    parameter spike_window = 50;
    
    parameter N_possibel_sp = 350000;
    
    input clk_d, clk_f;
    input rst;
    input signed [data_width - 1 : 0] input_data;
    input CS_idle;
    output spike_counter, wen_MEMspikelab, ren_MEMspikelab, ren_spikesegment, dout_spikesegment;
    
    (* ram_style = "block" *) reg signed [data_width - 1 : 0] MEM_200 [L_MEM_200 * ch_num - 1 : 0];
    reg wen_MEM200; //, ren_MEM200;
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] waddr_MEM200 = 0, raddr_MEM200_m = ch_num * 50; //, raddr_MEM200_l = ch_num * 49, , raddr_MEM200_r = ch_num * 51;

    
    always @(posedge clk_d) begin
        wen_MEM200 <= 1;
    end
    
    always @(posedge clk_d) begin
        if (wen_MEM200) begin
            if (waddr_MEM200 < L_MEM_200 * ch_num - 1) begin
                waddr_MEM200 <= waddr_MEM200 + 1;
            end
            else begin
                waddr_MEM200 <= 0;
            end
        end
        else begin
            waddr_MEM200 <= 0;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEM200) begin
            MEM_200 [waddr_MEM200] <= input_data;
        end
    end
    
    (* ram_style = "block" *) reg [3 * data_width - 1 : 0] MEM_3data [ch_num - 1 : 0];
    reg wen_MEM3data = 0, ren_MEM3data = 0;
    reg [$clog2(ch_num) - 1 : 0] waddr_MEM3data = 0, raddr_MEM3data = 0;
    wire [3 * data_width - 1 : 0] dout_MEM3data;
    
    reg [$clog2(ch_num * (spike_window + 2)) - 1 : 0] waddr_couner;
    always @(posedge clk_d) begin
        if (wen_MEM200) begin
            if (waddr_couner  < ch_num * (spike_window + 2) - 1) begin
                waddr_couner <= waddr_couner + 1;
            end
            else begin 
                 waddr_couner <= waddr_couner;
            end
        end
        else begin
            waddr_couner <= 0;
        end
    end
    
    always @(posedge clk_d) begin
        if (waddr_couner >= ch_num * (spike_window - 1) - 1) begin
            wen_MEM3data <= 1;
        end
        else begin
            wen_MEM3data <= 0;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEM3data) begin
            if (waddr_MEM3data < ch_num - 1) begin
                waddr_MEM3data <= waddr_MEM3data + 1;
            end
            else begin
                waddr_MEM3data <= 0;
            end
        end
        else begin
            waddr_MEM3data <= 0;
        end
    end
    
    reg [2 : 0] counter_loop = 0;
    always @(posedge clk_d) begin
        if (rst) begin
            counter_loop <= 0;
        end
        if (waddr_MEM3data == ch_num - 1) begin
            if (counter_loop < 2) begin
                counter_loop <= counter_loop + 1;
            end
            else begin
                counter_loop <= 0;
            end
        end
        else begin
            counter_loop <= counter_loop;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEM3data) begin
            MEM_3data[waddr_MEM3data][data_width * counter_loop +: data_width] <= input_data;
        end
    end
    
    
    always @(posedge clk_d) begin
        if (waddr_couner == ch_num * (spike_window + 2) - 1) begin
            ren_MEM3data <= 1;
        end
        else begin
            ren_MEM3data <= 0;
        end
    end
    
    
    
    always @(posedge clk_d) begin
        if (ren_MEM3data) begin
            if (raddr_MEM3data < L_MEM_200 - 1) begin
                raddr_MEM3data <= raddr_MEM3data + 1;
            end
            else begin
                raddr_MEM3data <= 0;
            end
        end
        else begin
            raddr_MEM3data <= 0;
        end
    end
    
    assign dout_MEM3data = (ren_MEM3data) ? MEM_3data[raddr_MEM3data] : 0;
    
    always @(posedge clk_d) begin
        if (ren_MEM3data) begin
            if (raddr_MEM200_m < ch_num * L_MEM_200 - 1) begin
                raddr_MEM200_m <= raddr_MEM200_m + 1;
            end
            else begin
                raddr_MEM200_m <= 0;
            end
        end
        else begin
            raddr_MEM200_m <= ch_num * 50;
        end
    end
    
//    wire [$clog2(ch_num * L_MEM_200) - 1 : 0] raddr_MEM200_l0, raddr_MEM200_r0;
    
//    assign raddr_MEM200_l0 = raddr_MEM200_m - ch_num;
//    assign raddr_MEM200_l = ((raddr_MEM200_l0 [$clog2(ch_num * L_MEM_200) - 1 : $clog2(ch_num * L_MEM_200) - 4]) >= 4'd8) ? (ch_num * L_MEM_200 + raddr_MEM200_l0) : raddr_MEM200_l0;
    
//    assign raddr_MEM200_r0 = raddr_MEM200_m + ch_num;
//    assign raddr_MEM200_r = (raddr_MEM200_r0 >= ch_num * L_MEM_200 - 1) ? (raddr_MEM200_r0 - ch_num * L_MEM_200) : raddr_MEM200_r0;
    reg signed [data_width - 1 : 0] dout_MEM200_l, dout_MEM200_m, dout_MEM200_r;
    always @(posedge clk_d) begin
        if (rst) begin
            dout_MEM200_l <= 0;
            dout_MEM200_m <= 0;
            dout_MEM200_r <= 0;
        end
        else if (ren_MEM3data) begin
            dout_MEM200_l <= dout_MEM3data[0 * data_width +: data_width];
            dout_MEM200_m <= dout_MEM3data[1 * data_width +: data_width];
            dout_MEM200_r <= dout_MEM3data[2 * data_width +: data_width];
        end
    end

    
    reg trig_sp;
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] spike_label = 0;
    
    always @(posedge clk_d) begin
        if ((dout_MEM200_m[data_width - 1 : data_width - 4] < 4'd8) && (dout_MEM200_m >= th) && (dout_MEM200_m > dout_MEM200_l) && (dout_MEM200_m > dout_MEM200_r)) begin
            trig_sp <= 1;
        end
        else begin
            trig_sp <= 0;
        end
    end
    
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] spike_label0;
    always @(posedge clk_d) begin
        spike_label0 <= raddr_MEM200_m;
    end
    always @(posedge clk_d) begin
        if (trig_sp) begin
            spike_label <= spike_label0;
        end
        else begin
            spike_label <= spike_label;
        end
    end
    
    reg [data_width - 1 : 0] spike_counter;
    always @(posedge clk_d) begin
        if (rst) begin
            spike_counter <= 0;
        end
        else if (trig_sp) begin
            spike_counter <= spike_counter + 1;
        end
        else begin
            spike_counter <= spike_counter;
        end
    end
    
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] MEM_spike_lab [N_load_spike - 1 : 0];
    reg wen_MEMspikelab = 0, ren_MEMspikelab = 0;
    reg [$clog2(N_load_spike) - 1 : 0] waddr_MEMspikelab = 0, raddr_MEMspikelab = 0;
    wire [$clog2(ch_num * L_MEM_200) - 1 : 0] dout_MEMspikelab;
    reg [$clog2(N_possibel_sp) - 1 : 0] w_spike_counter = 0, r_spike_counter = 0;
    
    always @(posedge clk_d) begin
        if (trig_sp) begin
            wen_MEMspikelab <= 1;
        end
        else begin
            wen_MEMspikelab <= 0;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEMspikelab) begin
            if (waddr_MEMspikelab < N_load_spike - 1) begin
                waddr_MEMspikelab <= waddr_MEMspikelab + 1;
            end
            else begin
                waddr_MEMspikelab <= 0;
            end
        end
        else begin
            waddr_MEMspikelab <= waddr_MEMspikelab;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEMspikelab) begin
            MEM_spike_lab [waddr_MEMspikelab] <= spike_label;
        end
    end
    
    always @(posedge clk_d) begin
        if (wen_MEMspikelab) begin
             w_spike_counter <= w_spike_counter + 1;
        end
        else begin
            w_spike_counter <= w_spike_counter;
        end
    end
    
    always @(posedge clk_f) begin
        if (CS_idle) begin
            ren_MEMspikelab <= 1;
        end
        else begin
            ren_MEMspikelab <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (ren_MEMspikelab && (r_spike_counter < w_spike_counter)) begin
            if (raddr_MEMspikelab < N_load_spike - 1) begin
                raddr_MEMspikelab <= raddr_MEMspikelab + 1;
            end
            else begin
                raddr_MEMspikelab <= 0;
            end
        end
        else begin
            raddr_MEMspikelab <= raddr_MEMspikelab;
        end
    end
    
    always @(posedge clk_f) begin
        if (ren_MEMspikelab && (r_spike_counter < w_spike_counter)) begin
            r_spike_counter <= r_spike_counter + 1;
        end
        else begin
            r_spike_counter <= r_spike_counter;
        end
    end
    
            
    assign dout_MEMspikelab = (ren_MEMspikelab) ? MEM_spike_lab[raddr_MEMspikelab] : dout_MEMspikelab;
    
//    wire [$clog2(ch_num * L_MEM_200) - 1 : 0] edge_max0, edge_min0;
//    assign edge_max0 = dout_MEMspikelab + spike_window * ch_num;
//    assign edge_min0 = dout_MEMspikelab - spike_window * ch_num;
    
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] edge_max = 0, edge_min = 0;
    
    always @(posedge clk_f) begin
        if (ren_MEMspikelab) begin
            if (dout_MEMspikelab + spike_window * ch_num > L_MEM_200 * ch_num - 1) begin
                edge_max <= dout_MEMspikelab + spike_window * ch_num - L_MEM_200 * ch_num;
            end
            else begin
                edge_max <= dout_MEMspikelab + spike_window * ch_num;
            end
        end
        else begin
            edge_max <= edge_max;
        end
    end
   
    always @(posedge clk_f) begin
        if (ren_MEMspikelab) begin
            if (dout_MEMspikelab < spike_window * ch_num) begin
                edge_min <= L_MEM_200 * ch_num - (spike_window * ch_num - dout_MEMspikelab);
            end
            else begin
                edge_min <= dout_MEMspikelab - spike_window * ch_num;
            end
        end
        else begin
            edge_min <= edge_min;
        end
    end
    reg ren_spikesegment0;
    always @(posedge clk_f) begin
        ren_spikesegment0 <= ren_MEMspikelab;
    end
    
    reg ren_spikesegment1 = 0, ren_spikesegment2;
    wire ren_spikesegment;
//    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] raddr_spikesegment0;
    reg [$clog2(ch_num * L_MEM_200) - 1 : 0] raddr_spikesegment = 0;
    wire signed [data_width - 1 : 0] dout_spikesegment;
    
    always @(posedge clk_f) begin
        if (ren_spikesegment0) begin
            ren_spikesegment1 <= 1;
        end
        else if (edge_max == edge_min) begin
            ren_spikesegment1 <= 0;
        end
        else if (edge_min < edge_max) begin
            if ((raddr_spikesegment < edge_max - ch_num) && (raddr_spikesegment >= edge_min)) begin
                ren_spikesegment1 <= 1;
            end
            else begin
                ren_spikesegment1 <= 0;
            end
        end
        else begin
            if (((raddr_spikesegment >= edge_min) && (raddr_spikesegment < L_MEM_200 * ch_num - 1)) || ((raddr_spikesegment < edge_max - ch_num) && (raddr_spikesegment >= 0))) begin
                ren_spikesegment1 <= 1;
            end
            else begin
                ren_spikesegment1 <= 0;
            end
        end            
    end
    
        
    always @(posedge clk_f) begin
        ren_spikesegment2 <= ren_spikesegment1;
    end
    
    assign ren_spikesegment = ren_spikesegment1 || ren_spikesegment2; 
//    always @(posedge clk_f) begin
//        if (rst) begin
//            ren_spikesegment <= 0;
//        end
//        else if (ren_MEMspikelab) begin
//            ren_spikesegment <= 1;
//        end
//        else if (ren_spikesegment0) begin
//            ren_spikesegment <= 1;
//        end
//        else begin
//            ren_spikesegment <= 0;
//        end
//    end
    
    always @(posedge clk_f) begin
        if (ren_spikesegment0) begin
            raddr_spikesegment <= edge_min;
        end
        else if (ren_spikesegment) begin
            if (raddr_spikesegment < L_MEM_200 * ch_num - 1) begin
                raddr_spikesegment <= raddr_spikesegment + ch_num;
            end
            else begin
                raddr_spikesegment <= raddr_spikesegment + ch_num - L_MEM_200 * ch_num;
            end
        end
        else begin
            raddr_spikesegment <= raddr_spikesegment;
        end
    end   
//    assign raddr_spikesegment = (raddr_spikesegment0 > L_MEM_200 * ch_num - 1) ? (raddr_spikesegment0 - L_MEM_200 * ch_num) : raddr_spikesegment0;
    
    assign dout_spikesegment = (ren_spikesegment) ? MEM_200[raddr_spikesegment] : 0;

    
      
            
    
endmodule
