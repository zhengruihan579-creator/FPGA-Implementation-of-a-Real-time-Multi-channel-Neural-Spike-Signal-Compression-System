`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/26 12:37:03
// Design Name: 
// Module Name: CS_MDC
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


module CS_MDC(
    clk_d, 
    clk_f,
    rst,
    EN_CS_idle,
    ren_spikesegment,
    dout_spikesegment, 
    CS_idle,
    ACC
    );
    
    parameter data_width = 16;
    parameter N_spike_segment = 100;
    parameter L_MEM_200 = 200;
    parameter ch_num = 384;
    parameter N_load_spike = 100;
    parameter spike_window = 50;
    parameter T_comp = 300;
    
    input clk_d, clk_f;
    input rst;
    input EN_CS_idle;
    input ren_spikesegment;
    input signed [data_width - 1 : 0] dout_spikesegment;
    output CS_idle,ACC;
    
    reg EN_MAX_MIN, EN_MAX_MIN1;
    always @(posedge clk_f) begin
        if (rst) begin
            EN_MAX_MIN <= 0;
            EN_MAX_MIN1 <= 0;
        end
        else begin
            EN_MAX_MIN <= ren_spikesegment;
            EN_MAX_MIN1 <= EN_MAX_MIN;
        end
    end
    
    reg signed [data_width - 1 : 0] sp_seg;
    always @(posedge clk_f) begin
        if (rst) begin
            sp_seg <= 0;
        end
        else if (ren_spikesegment) begin
            sp_seg <= dout_spikesegment;
        end
    end
    
    reg signed [data_width - 1 : 0] MAX, MIN;
    
    always @(posedge clk_f) begin
        if (rst) begin
            MAX <= 0;
        end
        else if (EN_MAX_MIN) begin
            if (sp_seg > MAX) begin
                MAX <= sp_seg;
            end
            else begin
                MAX <= MAX;
            end
        end
        else begin
            MAX <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            MIN <= 0;
        end
        else if (EN_MAX_MIN) begin
            if (sp_seg < MIN) begin
                MIN <= sp_seg;
            end
            else begin
                MIN <= MIN;
            end
        end
        else begin
            MIN <= 0;
        end
    end
    
//    reg signed [data_width - 1 : 0] data_range;
//    always @(posedge clk_f) begin
//        if (rst) begin
//            data_range <= 0;
//        end
//        else if ((~EN_MAX_MIN) && EN_MAX_MIN1) begin
//            data_range <= MAX - MIN;
//        end
//        else begin
//            data_range <= data_range;
//        end
//    end
    
//    reg [data_width - 1 : 0] M;
//    always @(posedge clk_f) begin
//        if (rst) begin
//            M <= 0;
//        end
//        else begin
//            M <= N_spike_segment - CR;
//        end
//    end
    
    reg [data_width - 1 : 0] sigma;
    always @(posedge clk_f) begin
        if (rst) begin
            sigma <= 0;
        end
        else if ((~EN_MAX_MIN) && EN_MAX_MIN1) begin
            sigma <= (MAX - MIN - 6 * ((MAX - MIN) >> 4)) >> 4;
        end
        else begin
            sigma <= sigma;
        end
    end
    
    reg [data_width * N_spike_segment - 1 : 0] reg_sp_seg;
    reg [$clog2(N_spike_segment) - 1 : 0] raddr_regspseg;
    always @(posedge clk_f) begin
        if (rst) begin
            raddr_regspseg <= 0;
        end
        else if (EN_MAX_MIN && (raddr_regspseg < N_spike_segment)) begin
            raddr_regspseg <= raddr_regspseg + 1;
        end
        else begin
            raddr_regspseg <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            reg_sp_seg <= 0;
        end
        else if (EN_MAX_MIN) begin
            reg_sp_seg[raddr_regspseg * data_width +: data_width] <= sp_seg;
        end
        else begin
            reg_sp_seg <= reg_sp_seg;
        end
    end
    
    reg sigma_done;
    always @(posedge clk_f) begin
        if (rst) begin
            sigma_done <= 0;
        end
        else if ((~ren_spikesegment) && EN_MAX_MIN) begin
            sigma_done <= 1;
        end
        else begin  
            sigma_done <= 0;
        end
    end
    
    reg [data_width * N_spike_segment - 1 : 0] SUB, Others;
    reg [N_spike_segment - 1 : 0] CLU_1b, CLU_1b_last;
    reg [data_width * 2 * N_spike_segment - 1 : 0] MUL;
    reg EN_SUB, EN_CLU;
    reg stop_clu, stop_comp, stop;
    reg flag0, flag1, flag2, flag3, flag4, flag5, flag6, flag7;
    
    reg signed [data_width - 1 : 0] core_data;
    reg [$clog2(N_spike_segment) - 1 : 0] core_data_addr;
    wire [data_width * 3 - 1 : 0] ACC;
    
    integer i;
    always @(posedge clk_f) begin
        if (rst) begin
            SUB <= 0;
        end
        else if (EN_SUB) begin
            for (i = 0; i < N_spike_segment; i = i + 1) begin
                if ((reg_sp_seg[data_width * (i + 1) - 4 +: 4] < 4'd8) && (core_data[data_width - 1] == 0)) begin
                    if ((reg_sp_seg[data_width * i +: data_width] >= core_data)) begin
                        SUB[data_width * i +: data_width] <= Others[data_width * i +: data_width] - core_data;
                    end
                    else begin
                        SUB[data_width * i +: data_width] <= core_data - reg_sp_seg[data_width * i +: data_width];
                    end
                end
                else if ((reg_sp_seg[data_width * (i + 1) - 4 +: 4] < 4'd8) && (core_data[data_width - 1])) begin
                    SUB[data_width * i +: data_width] <= ~core_data + 16'd1 + reg_sp_seg[data_width * i +: data_width];
                end
                else if ((reg_sp_seg[data_width * (i + 1) - 4 +: 4] >= 4'd8) && (core_data[data_width - 1] == 0)) begin
                    SUB[data_width * i +: data_width] <= ~reg_sp_seg[data_width * i +: data_width] + 16'd1 + core_data;
                end
                else begin
                    if (( ~reg_sp_seg[data_width * i +: data_width] + 16'd1 >= ~core_data + 16'd1)) begin
                        SUB[data_width * i +: data_width] <= ~reg_sp_seg[data_width * i +: data_width] + 16'd1 - (~core_data + 16'd1);
                    end
                    else begin
                        SUB[data_width * i +: data_width] <= ~core_data + 16'd1 - (~reg_sp_seg[data_width * i +: data_width] + 16'd1);
                    end
                end
            end
        end
        else if (stop_comp)begin
            SUB <= 0;
        end
        else begin
            SUB <= SUB;
        end
    end
    
    
    
    integer j;
    always @(posedge clk_f) begin
        if (rst) begin
            Others <= 0;
            CLU_1b <= 0;
            MUL <= 0;
        end
        else if (sigma_done) begin
            Others <= reg_sp_seg;
            CLU_1b <= 0;
            MUL <= 0;
        end
        else if (EN_CLU) begin
            for (j = 0; j < N_spike_segment; j = j + 1) begin
                if (CLU_1b_last[j] == 0) begin
                    if (SUB[data_width * j +: data_width] <= sigma) begin
                        Others[data_width * j +: data_width] <= 0;
                        CLU_1b[j] <= 1;
                        MUL[data_width * 2 * j +: data_width * 2] <= reg_sp_seg[data_width * j +: data_width] * reg_sp_seg[data_width * j +: data_width];
                    end
                    else begin
                        Others[data_width * j +: data_width] <= reg_sp_seg[data_width * j +: data_width];
                        CLU_1b[j] <= 0;
                        MUL[data_width * 2 * j +: data_width * 2] <= 0;
                    end
                end
            end
        end
        else if (stop_comp) begin
            Others <= 0;
            CLU_1b <= 0;
            MUL <= 0;
        end
        else begin
            Others <= Others;
            CLU_1b <= CLU_1b;
            MUL <= MUL;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            CLU_1b_last <= 0;
        end
        if (flag6) begin
            CLU_1b_last <= CLU_1b;
        end
        else begin
            CLU_1b_last <= CLU_1b_last;
        end
    end
      
    integer w;
    always @(posedge clk_f) begin
        if (rst) begin
            core_data_addr <= 0;
        end
        else if (flag4) begin
            for (w = N_spike_segment - 1; w >= 0; w = w - 1) begin
                if (CLU_1b[w] == 0) begin
                    core_data_addr <= w;
                end
            end
        end
        else begin
            core_data_addr <= core_data_addr;
        end
    end
    
    wire [3*data_width-1:0] acc_out1;
    Accumulator_100 acc1(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*10-1:data_width*2*0]),
        .acc_out(acc_out1)
    );  
    
    wire [3*data_width-1:0] acc_out2;
    Accumulator_100 acc2(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*20-1:data_width*2*10]),
        .acc_out(acc_out2)
    );  
    
    wire [3*data_width-1:0] acc_out3;
    Accumulator_100 acc3(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*30-1:data_width*2*20]),
        .acc_out(acc_out3)
    );  
    
    wire [3*data_width-1:0] acc_out4;
    Accumulator_100 acc4(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*40-1:data_width*2*30]),
        .acc_out(acc_out4)
    );  
    
    wire [3*data_width-1:0] acc_out5;
    Accumulator_100 acc5(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*50-1:data_width*2*40]),
        .acc_out(acc_out5)
    );  
    
    wire [3*data_width-1:0] acc_out6;
    Accumulator_100 acc6(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*60-1:data_width*2*50]),
        .acc_out(acc_out6)
    );  
    
    wire [3*data_width-1:0] acc_out7;
    Accumulator_100 acc7(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*70-1:data_width*2*60]),
        .acc_out(acc_out7)
    );  
    
    wire [3*data_width-1:0] acc_out8;
    Accumulator_100 acc8(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*80-1:data_width*2*70]),
        .acc_out(acc_out8)
    );  
    
    wire [3*data_width-1:0] acc_out9;
    Accumulator_100 acc9(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*90-1:data_width*2*80]),
        .acc_out(acc_out9)
    );  
    
    wire [3*data_width-1:0] acc_out10;
    Accumulator_100 acc10(
        .clock(clk_f),
        .rst(rst),
        .en(flag5),
        .data_in(MUL[data_width*2*100-1:data_width*2*90]),
        .acc_out(acc_out10)
    );  
    

    assign ACC = acc_out1 + acc_out2 + acc_out3 + acc_out4 + acc_out5 + acc_out6 + acc_out7 + acc_out8 + acc_out9 + acc_out10;
    
    always @(posedge clk_f) begin
        if (rst) begin
            core_data <= 0;
        end
        else if (flag0 || stop_clu) begin
            core_data <= Others[core_data_addr * data_width +: data_width];
        end
        else begin
            core_data <= core_data;
        end
    end
    
//    wire [$clog2(N_spike_segment) - 1 : 0] clu_num;
//    cal_1_num C1(
//        .clock(clk_f),
//        .access_port_r(CLU_1b),
//        .one_in_access_port_o(clu_num)
//    );
    
    reg comp_counter_flag;
    reg [$clog2(T_comp) - 1 : 0] comp_counter;
    
    always @(posedge clk_f) begin
        if (rst) begin
            comp_counter_flag <= 0;
        end
        else if (sigma_done) begin
            comp_counter_flag <= 1;
        end
        else if (comp_counter == T_comp - 2)begin
            comp_counter_flag <= 0;
        end
        else begin
            comp_counter_flag <= comp_counter_flag;
        end
    end
        
//    reg comp_counter_flag1;
//    always @(posedge clk_f) begin
//        if (rst) begin
//            comp_counter_flag1 <= 0;
//        end
//        else begin
//            comp_counter_flag1 <= comp_counter_flag;
//        end
//    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            comp_counter <= 0;
        end
        else if (comp_counter_flag && (comp_counter < T_comp - 2)) begin
            comp_counter <= comp_counter + 1;
        end
        else begin  
            comp_counter <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            stop_comp <= 0;
        end
        else if (comp_counter == T_comp - 2) begin
            stop_comp <= 1;
        end
        else begin
            stop_comp <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin
            stop <= 0;
        end
        else if (stop_comp) begin
            stop <= 1;
        end
        else if (raddr_regspseg == 63) begin
            stop <= 0;
        end
        else begin
            stop <= stop;
        end
    end
    
    reg CS_idle;
    always @(posedge clk_f) begin
        if (rst) begin
            CS_idle <= 0;
        end
        else if (EN_CS_idle || stop_comp) begin
            CS_idle <= 1;
        end
        else begin
            CS_idle <= 0;
        end
    end
   
    always @(posedge clk_f) begin
        if (rst) begin
            flag0 <= 0;
        end
        else if (sigma_done || stop_clu) begin
            if (stop) begin
                flag0 <= 0;
            end
            else begin
                flag0 <= 1;
            end          
        end
        else begin
            flag0 <= 0;
        end
    end
    
    always @(posedge clk_f) begin
        if (rst) begin  
            flag1 <= 0;
            EN_SUB <= 0;
            flag2 <= 0;
            flag3 <= 0;
            EN_CLU <= 0;
            flag4 <= 0;
            flag5 <= 0;
            flag6 <= 0;
            flag7 <= 0;
            stop_clu <= 0;
        end
        else begin
            flag1 <= flag0;
            EN_SUB <= flag1;
            flag2 <= EN_SUB;
            flag3 <= flag2;
            EN_CLU <= flag3;
            flag4 <= EN_CLU;
            flag5 <= flag4;
            flag6 <= flag5;
            flag7 <= flag6;
            stop_clu <= flag7;
        end
    end
    
    
    
    
endmodule

