#!/bin/bash
source config.env

N_GPUS=$(nvidia-smi --list-gpus | wc -l)

declare -A GPU_UTILS
declare -A MEM_UTILS

POLL=1

while true
do
  CTS=$(date +%Y%m%d-%H%M%S)

  for ((i = 0 ; i < N_GPUS ; i++ ))
  do
    GPU_UTILS[$i]=0
    MEM_UTILS[$i]=0
  done

  N_GPU_LOW=0
  N_MEM_LOW=0

  echo "[$CTS] #$POLL polling..."
  # echo "⌛ sampling ..."
  for i in $(seq 1 $N_SAMPLE)
  do
    SAMPLE=$(nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,nounits,noheader)
    # echo "📏 taking sample-$i"
    while IFS=, read -r index gpu mem; do
      ((GPU_UTILS[$index]+=$gpu))
      ((MEM_UTILS[$index]+=$mem))
    done <<< $SAMPLE

    sleep $SAMPLE_INTERVAL
  done
  # echo "✅ sampling done"

  CONTENT="**DGX-A100 GPU Utilization**"
  CONTENT+='\n'
  CONTENT+='```\n'
  CONTENT+='------------------------\n'
  CONTENT+='|  N |   GPU% |   MEM% |\n'
  CONTENT+='------------------------\n'
  for ((i = 0 ; i < N_GPUS ; i++ ))
  do
    ((GPU_UTILS[$i]/=N_SAMPLE))
    ((MEM_UTILS[$i]/=N_SAMPLE))

    GPU_STATUS="🟥"
    if [[ ${GPU_UTILS[$i]} -lt $LIMIT_MED ]]; then GPU_STATUS="🟧"; fi
    if [[ ${GPU_UTILS[$i]} -lt $LIMIT_LOW ]]
    then
      GPU_STATUS="🟩"
      ((N_GPU_LOW++))
    fi

    MEM_STATUS="🟥"
    if [[ ${MEM_UTILS[$i]} -lt $LIMIT_MED ]]; then MEM_STATUS="🟧"; fi
    if [[ ${MEM_UTILS[$i]} -lt $LIMIT_LOW ]]
    then
      MEM_STATUS="🟩"
      ((N_MEM_LOW++))
    fi

    CONTENT+=$(printf "| %2s | ${GPU_STATUS} %3s | ${MEM_STATUS} %3s |" $i ${GPU_UTILS[$i]} ${MEM_UTILS[$i]})
    CONTENT+='\n'
  done
  CONTENT+='------------------------\n'
  CONTENT+='```'
  DATA="{\"content\": \"$CONTENT\"}"

  # echo ""

  # echo -e "$CONTENT"
  CTS=$(date +%Y%m%d-%H%M%S)
  echo "[$CTS] #$POLL GPU: ${GPU_UTILS[@]}"
  CTS=$(date +%Y%m%d-%H%M%S)
  echo "[$CTS] #$POLL MEM: ${MEM_UTILS[@]}"
  
  if [[ $N_GPU_LOW -gt 0 ]]
  then
    CTS=$(date +%Y%m%d-%H%M%S)
    echo "[$CTS] #$POLL found $N_GPU_LOW usable GPU(s), reporting..."
    wget --no-check-certificate --quiet --method POST --header 'Content-Type: application/json' --body-data "$DATA" "$WEBHOOK_URL"
  fi

  # echo ""
  # echo "⌛ waiting ${POLL_INTERVAL}s until next poll..."

  ((POLL++))
  sleep $POLL_INTERVAL
done;