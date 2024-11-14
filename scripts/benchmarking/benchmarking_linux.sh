# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Customize what is being run
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DRY_RUN=0

RUN_CUDA_EAGER=0
RUN_CUDA_COMPILE=0
RUN_CUDA_AOTI=0
RUN_CUDA_AOTI_PT2=0

RUN_CPU_EAGER=0
RUN_CPU_COMPILE=0
RUN_CPU_AOTI=1
RUN_CPU_AOTI_PT2=1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check and Set Up Args (model, out_directory)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if [ $# -ne 2 ]; then
  echo "Please provide (1) model and (2) directory as positional arguments"
  exit 1
fi

model=$1
dir=$2

mkdir -p $dir


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Env Variables for Running Commands
ENV_VARIABLE="OMP_NUM_THREADS=16 numactl --cpunodebind=0 --membind=0"

# Function for printing and writing to files
function formatted_export_and_generate {
  local file="$dir/$1"
  local generate_cmd="${ENV_VARIABLE} $2"
  local compile_cmd="$3"

  # Write Commands to the top of the output file
  echo $compile_cmd > $file
  echo $generate_cmd >> $file

  echo "Writing to: ${file}"

  # Export the Model
  if [ ! -z "$compile_cmd" ]; then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> $file
    echo "$compile_cmd" | tee -a $file
    if [ $DRY_RUN -eq 0 ]; then
      eval $compile_cmd &>> $file
    fi
  fi

  # Generate using the Model
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> $file
  echo $generate_cmd | tee -a $file
    if [ $DRY_RUN -eq 0 ]; then
      eval $generate_cmd &>> $file
    fi
  echo
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cuda eager
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CUDA_EAGER -eq 1 ]; then
  echo "Cuda eager b16"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --num-samples 3"
  file="cuda_eager_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "Cuda eager int8"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --num-samples 3"
  file="cuda_eager_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "Cuda eager int4"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --num-samples 3"
  file="cuda_eager_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cuda compile
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CUDA_COMPILE -eq 1 ]; then
  echo "Cuda compile b16"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --compile --num-samples 3"
  file="cuda_compile_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "Cuda compile int8"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --compile --num-samples 3"
  file="cuda_compile_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "Cuda compile int4"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --prompt \"Once upon a time,\" --max-new-tokens 200 --compile --num-samples 3"
  file="cuda_compile_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU eager
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CPU_EAGER -eq 1 ]; then
  echo "CPU eager b16"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --num-samples 3"
  file="cpu_eager_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "CPU eager int8"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --num-samples 3"
  file="cpu_eager_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "CPU eager int4"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --num-samples 3"
  file="cpu_eager_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU compile
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CPU_COMPILE -eq 1 ]; then
  echo "CPU compile b16"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --compile --num-samples 3"
  file="cpu_compile_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "CPU compile int8"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --compile --num-samples 3"
  file="cpu_compile_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd"

  echo "CPU compile int4"
  generate_cmd="python3 torchchat.py generate $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --prompt \"Once upon a time,\" --max-new-tokens 256 --compile --num-samples 3"
  file="cpu_compile_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cuda AOTI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CUDA_AOTI -eq 1 ]; then
  echo "Cuda aoti b16"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-dso-path /tmp/model16.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model16.so --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "Cuda aoti int8"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-dso-path /tmp/model8.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model8.so --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "Cuda aoti int4"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-dso-path /tmp/model34.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model34.so --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cuda AOTI PT2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CUDA_AOTI_PT2 -eq 1 ]; then
  echo "Cuda aoti PT2 b16"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-aoti-package-path /tmp/model16.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model16.pt2 --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_pt2_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "Cuda aoti PT2 int8"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-aoti-package-path /tmp/model8.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model8.pt2 --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_pt2_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "Cuda aoti PT2 int4"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cuda\"}}' --output-aoti-package-path /tmp/model34.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model34.pt2 --prompt \"Once upon a time,\" --max-new-tokens 200 --device cuda --num-samples 3"
  file="cuda_aoti_pt2_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU AOTI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CPU_AOTI -eq 1 ]; then
  echo "CPU aoti b16"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-dso-path /tmp/model16.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model16.so --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "CPU aoti int8"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-dso-path /tmp/model8.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model8.so --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "CPU aoti int4"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-dso-path /tmp/model34.so"
  generate_cmd="python3 torchchat.py generate $model --dso-path /tmp/model34.so --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU AOTI PT2
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ $RUN_CPU_AOTI_PT2 -eq 1 ]; then
  echo "CPU aoti PT2 b16"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-aoti-package-path /tmp/model16.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model16.pt2 --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_pt2_b16.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "CPU aoti PT2 int8"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int8\": {\"groupsize\": 0}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-aoti-package-path /tmp/model8.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model8.pt2 --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_pt2_8.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"

  echo "CPU aoti PT2 int4"
  compile_cmd="python3 torchchat.py export $model --quantize '{\"linear:int4\": {\"groupsize\": 256}, \"precision\": {\"dtype\":\"bfloat16\"}, \"executor\":{\"accelerator\":\"cpu\"}}' --output-aoti-package-path /tmp/model34.pt2"
  generate_cmd="python3 torchchat.py generate $model --aoti-package-path /tmp/model34.pt2 --prompt \"Once upon a time,\" --max-new-tokens 256 --device cpu --num-samples 3"
  file="cpu_aoti_pt2_4.txt"
  formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd"
fi
