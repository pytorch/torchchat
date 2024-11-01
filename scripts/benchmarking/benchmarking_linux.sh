
RUN_CUDA_EAGER=true
RUN_CUDA_COMPILE=true
RUN_CUDA_AOTI=true

RUN_CPU_EAGER=true
RUN_CPU_COMPILE=true
RUN_CPU_AOTI=true

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

# Function for printing and writing to files
function formatted_export_and_generate {
  local file="$dir/$1"
  local generate_cmd="$2"
  local compile_cmd="$3"

  # Write Commands to the top of the output file
  echo $compile_cmd > $file
  echo $generate_cmd >> $file

  echo "Writing to: ${file}"

  # Export the Model
  if [ ! -z "$compile_cmd" ]; then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> $file
    echo "$compile_cmd" | tee -a $file
    eval $compile_cmd &>> $file 
  fi

  # Generate using the Model
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> $file
  echo $generate_cmd | tee -a $file
  eval $generate_cmd &>> $file
  echo
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cuda eager
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
if [ "$RUN_CUDA_EAGER" = "true" ]; then
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

if [ "$RUN_CUDA_COMPILE" = "true" ]; then
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
  # formatted_export_and_generate "$file" "$generate_cmd"
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU eager 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ "$RUN_CPU_EAGER" = "true" ]; then
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

if [ "$RUN_CPU_COMPILE" = "true" ]; then
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

if [ "$RUN_CUDA_AOTI" = "true" ]; then
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
  # formatted_export_and_generate "$file" "$generate_cmd" "$compile_cmd" 
fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CPU AOTI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if [ "$RUN_CPU_AOTI" = "true" ]; then
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
