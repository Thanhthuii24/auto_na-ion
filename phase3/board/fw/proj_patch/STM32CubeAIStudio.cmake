# Edited for the v5 streaming app (phase3/board/fw): two ST C-models (enc, head) + naion app, no validation stack.
set(NA ${CMAKE_CURRENT_LIST_DIR}/../..)
target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../Middlewares/ST/AI/Lib)
target_link_libraries(${CMAKE_PROJECT_NAME} :NetworkRuntime1210_CM7_GCC.a m)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../Middlewares/ST/AI/Misc/Src/syscalls.c
    ${CMAKE_CURRENT_LIST_DIR}/../Middlewares/ST/AI/Misc/Src/aiTestUtility.c
    ${CMAKE_CURRENT_LIST_DIR}/AI/App/user_init.c
    ${NA}/net_enc/enc.c ${NA}/net_enc/enc_data.c
    ${NA}/net_head/head.c ${NA}/net_head/head_data.c
    ${NA}/naion_app.c ${NA}/naion_data.c ${NA}/app_naion.c)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../Middlewares/ST/AI/Misc/Inc ${CMAKE_CURRENT_LIST_DIR}/../Middlewares/ST/AI/Inc
    ${CMAKE_CURRENT_LIST_DIR}/AI/App ${NA} ${NA}/net_enc ${NA}/net_head)
target_sources(STM32_Drivers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_xspi.c
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/BSP/STM32H7RSxx_Nucleo/stm32h7rsxx_nucleo.c
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/BSP/Components/mx25uw25645g/mx25uw25645g.c)
target_include_directories(STM32_Drivers PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/STM32H7RSxx_HAL_Driver/Inc
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/BSP/Components/mx25uw25645g
    ${CMAKE_CURRENT_LIST_DIR}/../Drivers/BSP/STM32H7RSxx_Nucleo)
