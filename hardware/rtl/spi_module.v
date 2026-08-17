`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/08/22 15:56:20
// Design Name: 
// Module Name: spi_module
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
//spi slaver rtx
module spi_module(
    sys_clk,
    rst,
    spi_cs,
    spi_clk,
    spi_mosi,
    rx_data_valid,
    rx_data
);
    
    parameter                                           BITS_LEN    = 16;    //????bit??
    parameter                                           CPOL    = 1'b0;    //????
    parameter                                           CPHA    = 1'b0;     //????
    
    
    input                                               sys_clk ;
    input                                               rst   ;
    input                                               spi_cs  ;          //spi????
    input                                               spi_clk ;           //spi_clk
    input                                               spi_mosi    ;      //spi_mosi
    output  reg                                         rx_data_valid   ;   //??????????
    output  reg     signed [BITS_LEN - 1:0]             rx_data    ;         //??????
    
    reg             [3:0]                               spi_cs_reg  ;       //???
    reg             [3:0]                               spi_clk_reg ;       //???
    reg             [3:0]                               spi_mosi_reg    ;   //???
    reg                                                 cap ;               //??????
    reg                                                 spi_clk_pos ;       //???
    reg                                                 spi_clk_neg ;       //???
    wire                                                rx_en   ;           //??????
    reg             [4:0]                               rx_bit_cnt  ;       //??bit???

    assign rx_en  = (~spi_cs_reg[3]);   
    
    //? cs?spi_clk,mosi???????????
    always @(posedge sys_clk) begin
        if(rst)
            spi_cs_reg <= 3'd0;
        else
            spi_cs_reg <= {spi_cs_reg[2:0],spi_cs};
    end
    
    always @(posedge sys_clk) begin
        if(rst)
            spi_clk_reg <= 3'd0;
        else
            spi_clk_reg <= {spi_clk_reg[2:0],spi_clk};
    end
    always @(posedge sys_clk) begin
        if(rst)
            spi_mosi_reg <= 3'd0;
        else
            spi_mosi_reg <= {spi_mosi_reg[2:0],spi_mosi};
    end
    
    //spi_clk???
    always @(posedge sys_clk) begin
        if(rst)
            spi_clk_pos <= 1'b0;
        else if (spi_clk_reg[2] == 1'b0 && spi_clk_reg[1] == 1'b1)
            spi_clk_pos <= 1'b1;
        else
            spi_clk_pos <= 1'b0;
    end
    
    //spi_clk???
    always @(posedge sys_clk) begin
        if(rst)
            spi_clk_neg <= 1'b0;
        else if (spi_clk_reg[2] == 1'b1 && spi_clk_reg[1] == 1'b0)
            spi_clk_neg <= 1'b1;
        else
            spi_clk_neg <= 1'b0;
    end
    
    //??CPOL CPHA?????????
    always @(posedge sys_clk) begin
        if(rst)
            cap <= 1'b0;
        else if(CPOL == 1'b0 && CPHA == 1'b0)
            cap <= spi_clk_pos;
        else if(CPOL == 1'b0 && CPHA == 1'b1)
            cap <= spi_clk_neg;
        else if(CPOL == 1'b1 && CPHA == 1'b0)
            cap <= spi_clk_neg;
        else if(CPOL == 1'b1 && CPHA == 1'b1)
            cap <= spi_clk_pos;
        else
            cap <= 1'b0;
    end
    
    //?????????bit???????
    always @(posedge sys_clk) begin
        if(rst)begin
            rx_bit_cnt <= 'd0;  
            rx_data_valid <= 1'b0;
        end
        else if((rx_en == 1'b1) && (cap == 1'b1) && (rx_bit_cnt < BITS_LEN))begin
            rx_bit_cnt <= rx_bit_cnt +1'b1;
            rx_data_valid <= 1'b0;
        end
        else if((rx_en == 1'b0) || (rx_bit_cnt == BITS_LEN))begin
            rx_bit_cnt <= 'd0;
            rx_data_valid <= 1'b1;
        end
        else begin
            rx_bit_cnt <= rx_bit_cnt;
            rx_data_valid <= 1'b0;
        end
    end
    
    //???????????????
    always @(posedge sys_clk) begin
        if(rst)
            rx_data <= 'd0;
        else if(rx_en == 1'b1 && cap == 1'b1)
            rx_data <= {rx_data[BITS_LEN -2 : 0],spi_mosi_reg[3]};
        else if(rx_en == 1'b0)
            rx_data <= 'd0;
        else
            rx_data <= rx_data;
    end

endmodule

