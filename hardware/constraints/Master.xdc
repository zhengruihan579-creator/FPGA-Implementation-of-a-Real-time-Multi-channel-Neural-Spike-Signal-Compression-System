#create_clock -period 8.000 -name sys_clk [get_ports sys_clk]
#set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets I_spi_clk_IBUF_inst/O]
#set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets rst_IBUF_inst/O]


#RLD3 100MHz clkp, clkn
set_property PACKAGE_PIN L22 [get_ports clk_n]
set_property IOSTANDARD LVDS [get_ports clk_n]

set_property PACKAGE_PIN L23 [get_ports clk_p]
set_property IOSTANDARD LVDS [get_ports clk_p]


#rst
set_property PACKAGE_PIN N36 [get_ports rst]
set_property IOSTANDARD LVCMOS18 [get_ports rst]

#set_property PACKAGE_PIN N37 [get_ports rst_cnt]
#set_property IOSTANDARD LVCMOS18 [get_ports rst_cnt]

#LED
#set_property PACKAGE_PIN AW15 [get_ports LED_flash[0]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[0]]

#set_property PACKAGE_PIN AV16 [get_ports LED_flash[1]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[1]]

#set_property PACKAGE_PIN BA15 [get_ports LED_flash[2]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[2]]

#set_property PACKAGE_PIN AY15 [get_ports LED_flash[3]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[3]]

#set_property PACKAGE_PIN AV17 [get_ports LED_flash[4]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[4]]

#set_property PACKAGE_PIN AU17 [get_ports LED_flash[5]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[5]]

#set_property PACKAGE_PIN AY16 [get_ports LED_flash[6]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[6]]

#set_property PACKAGE_PIN AW16 [get_ports LED_flash[7]]
#set_property IOSTANDARD LVCMOS18 [get_ports LED_flash[7]]

##spi
#set_property PACKAGE_PIN C23 [get_ports I_spi_cs]
#set_property IOSTANDARD LVCMOS12 [get_ports I_spi_cs]

#set_property PACKAGE_PIN E22 [get_ports I_spi_clk]
#set_property IOSTANDARD LVCMOS12 [get_ports I_spi_clk]

#set_property PACKAGE_PIN F22 [get_ports I_spi_mosi]
#set_property IOSTANDARD LVCMOS12 [get_ports I_spi_mosi]

set_property PACKAGE_PIN A23 [get_ports ACC[0]]
set_property IOSTANDARD LVCMOS12 [get_ports ACC[0]]

#set_property PACKAGE_PIN A24 [get_ports I_spi_mosi[2]]
#set_property IOSTANDARD LVCMOS12 [get_ports I_spi_mosi[2]]

#set_property PACKAGE_PIN B24 [get_ports I_spi_mosi[3]]
#set_property IOSTANDARD LVCMOS12 [get_ports I_spi_mosi[3]]


set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]

set_property BITSTREAM.General.UnconstrainedPins {Allow} [current_design]