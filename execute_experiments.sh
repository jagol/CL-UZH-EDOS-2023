
# 0-single-task
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_A_LD_false_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_A_LD_false_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_A_LD_false_DI_false_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_B_LD_false_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_B_LD_false_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_B_LD_false_DI_false_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_C_LD_false_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_C_LD_false_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/0-single-task/EDOS_C_LD_false_DI_false_run3.json

# 1-single-task
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_A_LD_true_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_A_LD_true_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_A_LD_true_DI_false_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_B_LD_true_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_B_LD_true_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_B_LD_true_DI_false_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_C_LD_true_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_C_LD_true_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/EDOS_C_LD_true_DI_false_run3.json

# 2-EDOS-multi-task-with-without-NLI
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_NLI_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_NLI_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/2-EDOS-multi-task-with-without-NLI/EDOS_ABC_LD_true_DI_false_NLI_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_A_LD_true_DI_false_NLI_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_A_LD_true_DI_false_NLI_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_A_LD_true_DI_false_NLI_run3.json

# 3-EDOS-multi-task-2-phases
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_B_LD_true_DI_false_NLI_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_B_LD_true_DI_false_NLI_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_B_LD_true_DI_false_NLI_run3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_C_LD_true_DI_false_NLI_run1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_C_LD_true_DI_false_NLI_run2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/3-EDOS-multi-task-2-phases/ph1_EDOS_ABC_ph2_EDOS_C_LD_true_DI_false_NLI_run3.json

# 4-EDOS-multi-task-3-phases
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run1_part_1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run2_part_1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run3_part_1.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run1_part_2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run2_part_2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/4-EDOS-multi-task-3-phases/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_run3_part_2.json

# 5-EDOS-multi-task-3-phases-DI
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run1_part_1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run1_part_2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run1_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run2_part_1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run2_part_2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run2_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run3_part_1.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run3_part_2.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_A_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/5-EDOS-multi-task-3-phases-DI/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_DI_run3_part_3.json

# 6-balancing-task-B
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_20p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_20p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_20p_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_25p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_25p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_25p_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_30p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_30p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-B/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_B_30p_DI_run3_part_3.json

# 6-balancing-task-C
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_20p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_20p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_20p_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_25p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_25p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_25p_DI_run3_part_3.json

CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_30p_DI_run1_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_30p_DI_run2_part_3.json
CUDA_VISIBLE_DEVICES=0 python3 src/exec_scripts.py -p configs/6-balancing-task-C/ph1_AUX_EDOS_ABC_ph2_EDOS_ABC_ph3_EDOS_C_30p_DI_run3_part_3.json
