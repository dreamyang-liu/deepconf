# set -ex
# for rid in {0..0}; do
#     for budget in 256 512; do
#         for qid in {0..29}; do
#             if [ -f "./outputs/deepconf_qid${qid}_rid${rid}"* ]; then
#                 echo "File deepconf_qid${qid}_rid${rid} already exists, skipping."
#                 continue
#             fi
#             # python deepconf-baseline.py --qid $qid --rid $rid
#             python deepconf-online.py --qid $qid --rid $rid --budget $budget
#         done
#     done
# done



set -ex
for rid in {0..0}; do
    for budget in 512; do
        for qid in {0..29}; do
            if [ -f "./outputs/deepconf_simple_qid${qid}_rid${rid}"* ]; then
                echo "File deepconf_simple_qid${qid}_rid${rid} already exists, skipping."
                continue
            fi
            # python deepconf-baseline.py --qid $qid --rid $rid
            python deepconf-offline.py --qid $qid --rid $rid
        done
    done
done
