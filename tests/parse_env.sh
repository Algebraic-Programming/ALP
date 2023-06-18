# do NOT run, just source

if [[ $_ = $0 ]]; then
	echo -e "you are running me, while you should just source me"
	exit -1
fi

SCRIPT_NAME="$0"

# Default modes
MODES="ndebug debug"

function print_synopsis() {
	echo "SYNOPSIS: ${SCRIPT_NAME} [OPTIONS] <file(s)...>"
	echo " OPTIONS:"
	echo "  --help prints this help"
	echo "  --backends (space-separated) list of backends to run the tests against"
	echo "  --test-bin-dir directory storing the tests"
	echo "  --test-out-dir directory to store the output of tests"
	echo "  --input-dir directory for input datasets"
	echo "  --lpfexe path to LPF runner"
	echo "  --lpf-engine LPF engine to run against"
	echo "  --lpf-args additional arguments for the LPF engine"
	echo "  --gnn-dataset-path "
	echo "  --manual-run-args LPF engine arguments for manual run"
	echo "  --output-verification-dir directory with golden output for test verification"
	echo "  --test-data-dir directory with original data used for tests"
	echo "  --modes (space-separated) modes for the unit-tests, default: $MODES"
}

function check_dir() {
	if [[ ! -d "$1" ]]; then
		echo -e "'$1' is not a directory"
		exit -1
	fi
}

cmd_args="$@"

while test -n "$1"
do
	case "$1" in
		--help|-h)
			print_synopsis
			exit 0
			;;
		--backends)
			BACKENDS=("$2")
			shift 2
			;;
		--test-bin-dir)
			TEST_BIN_DIR="$2"
			check_dir "${TEST_BIN_DIR}"
			TEST_BIN_DIR="$( cd "$2" &> /dev/null && pwd )"
			shift 2
			;;
		--test-out-dir)
			TEST_OUT_DIR="$2"
			shift 2
			;;
		--input-dir)
			INPUT_DIR="$( cd "$2" &> /dev/null && pwd )"
			if [[ "$?" != "0" ]]; then
				# print in yellow
				echo -e "\033[1;33m>> '$2' is an invalid path: tests depending on a dataset will be skipped\033[0m"
				INPUT_DIR=''
			fi
			shift 2
			;;
		--gnn-dataset-path)
			GNN_DATASET_PATH="$2"
			if [[ ! -d "${GNN_DATASET_PATH}" ]]; then
				echo -e "${GNN_DATASET_PATH} is not a valid directory"
				exit -1
			fi
			GNN_DATASET_PATH="$(realpath -e "$2")"
			shift 2
			;;
		--lpfexe)
			LPFEXE="$2"
			if [[ ! -x "${LPFEXE}" ]]; then
				echo -e "${LPEXE} is not executable or does not exist"
				exit -1
			fi
			LPFEXE="$(realpath "$2")"
			shift 2
			;;
		--lpf-engine)
			LPF_ENGINE="$2"
			shift 2
			;;
		--lpf-args)
			LPF_ARGS="$2"
			if [[ -z "${LPF_ARGS}" ]]; then
					echo -e "--lpf-args cannot be empty"
					exit -1
			fi
			shift 2
			;;
		--manual-run-args)
			MANUAL_RUN_ARGS="$2"
			if [[ -z "${MANUAL_RUN_ARGS}" ]]; then
				echo -e "--manual-run-args cannot be empty"
				exit -1
			fi
			shift 2
			;;
		--test-data-dir)
			TEST_DATA_DIR="$(realpath "$2")"
			shift 2
			;;
		--output-verification-dir)
			OUTPUT_VERIFICATION_DIR="$(realpath "$2")"
			shift 2
			;;
		--modes)
			MODES=("$2")
			shift 2
			;;
		--*)
			echo -e "unknown option '$1' inside"
			echo "---"
			echo "${cmd_args}"
			echo "---"
			print_synopsis
			exit -1
			;;
		*)
			break
			;;
	esac
done

# some parameters are mandatory
if [[ -z "${TEST_BIN_DIR}" ]]; then
	echo "no argument for --test-bin-dir"
	exit 1
fi
if [[ -z "${TEST_OUT_DIR}" ]]; then
	echo "no argument for --test-out-dir"
	exit 1
fi
if [[ -z "${BACKENDS}" ]]; then
	echo "no argument for --backends"
	exit 1
fi
if [[ -z "${MODES}" ]]; then
	echo "no argument for --modes"
	exit 1
fi

# print in green
function print_green() {
	local text="$@"
	echo -e "\033[1;32m"${text}"\033[0m"
}
print_green ">>>   RUNNING TESTS IN ${SCRIPT_NAME}   <<<"

mkdir ${TEST_OUT_DIR} || true
TEST_OUT_DIR=$(realpath "${TEST_OUT_DIR}")

# define LPFRUN only if it is not already defined AND LPFEXE is defined
# if none of these variables is defined, it means we are not going to run LPF benchmarks anyway
if [[ -z "${LPFRUN}" && ! -z "${LPFEXE}" ]]; then

	# command to run LPF programs
	if [[ -z "${LPF_ENGINE}" ]]; then
		LPFRUN="${LPFEXE} ${LPF_ARGS}"
	else
		LPFRUN="${LPFEXE} ${LPF_ARGS} -engine ${LPF_ENGINE}"
	fi

fi

# similar strategy for MANUALRUN
if [[ -z "${MANUALRUN}" && ! -z "${LPFEXE}" ]]; then
	MANUALRUN="${LPFRUN} ${MANUAL_RUN_ARGS}"
fi

# in case LPF is enabled, define some additional environment variables
if [[ ! -z "${LPFRUN}" ]]; then

	# switch to pass arguments to any underlying runner
	if [ -z "${LPFRUN_PASSTHROUGH}" ]; then
		LPFRUN_PASSTHROUGH="-mpirun,"
		echo "Warning: LPFRUN_PASSTHROUGH was not set. I assumed the following: -mpirun,"
	fi

	# switch to pass environment variables to the underlying MPI layer
	if [ -z "${MPI_PASS_ENV}" ]; then
		# MPICH / Intel MPI:
		MPI_PASS_ENV=${LPFRUN_PASSTHROUGH}-genv
		# OpenMPI
		#MPI_PASS_ENV=${LPFRUN_PASSTHROUGH}-x
		# IBM Platform MPI
		#MPI_PASS_ENV=${LPFRUN_PASSTHROUGH}-e
		echo "Warning: MPI_PASS_ENV was not set. I assumed the following: ${MPI_PASS_ENV}"
	fi

	# the following two variables are used by both unit and smoke tests, and hence defined here
	if [[ -z ${BIND_PROCESSES_TO_HW_THREADS} ]]; then
		# MPICH and OpenMPI
		BIND_PROCESSES_TO_HW_THREADS="${LPFRUN_PASSTHROUGH}-bind-to ${LPFRUN_PASSTHROUGH}hwthread"
		# Intel MPI
		#BIND_PROCESSES_TO_HW_THREADS="${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN=1 ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_DOMAIN=core"
		# IBM Platform MPI
		#BIND_PROCESSES_TO_HW_THREADS="${LPFRUN_PASSTHROUGH}-affcycle=numa ${LPFRUN_PASSTHROUGH}-affwidth=core"
		printf "Warning: BIND_PROCESSES_TO_HW_THREADS environment variable was not set. "
		printf "I assumed the following: ${BIND_PROCESSES_TO_HW_THREADS}\n"
	fi
	if [[ -z ${BIND_PROCESSES_TO_MULTIPLE_HW_THREADS} ]]; then
		# MPICH
		BIND_PROCESSES_TO_MULTIPLE_HW_THREADS="${LPFRUN_PASSTHROUGH}-bind-to ${LPFRUN_PASSTHROUGH}hwthread:"
		# OpenMPI
		#BIND_PROCESSES_TO_MULTIPLE_HW_THREADS="${LPFRUN_PASSTHROUGH}--map-by ${LPFRUN_PASSTHROUGH}socket ${LPFRUN_PASSTHROUGH}--bind-to ${LPFRUN_PASSTHROUGH}socket ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}IGNORE_NUMBER_OF_THREADS="
		# Intel MPI
		#BIND_PROCESSES_TO_HW_THREADS="${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}I_MPI_PIN=1 ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}I_MPI_PIN_DOMAIN="
		# IBM Platform MPI
		#BIND_PROCESSES_TO_HW_THREADS="${LPFRUN_PASSTHROUGH}-affcycle=numa ${LPFRUN_PASSTHROUGH}-affwidth=core ${LPFRUN_PASSTHROUGH}-affblock="
		printf "Warning: BIND_PROCESSES_TO_MULTIPLE_HW_THREADS environment variable "
		printf "was not set. I assumed the following: "
		echo "${BIND_PROCESSES_TO_MULTIPLE_HW_THREADS}<T>"
	fi

fi

if [[ -z ${MAX_THREADS} ]]; then
	if ! command -v nproc &> /dev/null; then
		echo "Error: nproc command does not exist while MAX_THREADS was not set."
		echo "Please set MAX_THREADS explicitly and try again."
		exit 255;
	else
		MAX_THREADS=`nproc --all`
		echo "Info: detected ${MAX_THREADS} threads"
	fi
fi


echo
echo "*** ENVIRONMENT ***"
echo " TEST_BIN_DIR=${TEST_BIN_DIR}"
echo " TEST_OUT_DIR=${TEST_OUT_DIR}"
echo " INPUT_DIR=${INPUT_DIR}"
echo " BACKENDS=${BACKENDS}"
echo " GNN_DATASET_PATH=${GNN_DATASET_PATH}"
echo " OUTPUT_VERIFICATION_DIR=${OUTPUT_VERIFICATION_DIR}"
echo " TEST_DATA_DIR=${TEST_DATA_DIR}"
if [[ ! -z "${LPFRUN}" ]]; then
	echo " LPFRUN=${LPFRUN}"
	echo " MANUALRUN=${MANUALRUN}"
	echo " LPFRUN_PASSTHROUGH=${LPFRUN_PASSTHROUGH}"
	echo " MPI_PASS_ENV=${MPI_PASS_ENV}"
	echo " BIND_PROCESSES_TO_HW_THREADS=${BIND_PROCESSES_TO_HW_THREADS}"
	echo " BIND_PROCESSES_TO_MULTIPLE_HW_THREADS=${BIND_PROCESSES_TO_MULTIPLE_HW_THREADS}"
fi
echo "*******************"
echo

# common parameters to check

if [[ -z "${BACKENDS}" ]]; then
	echo "BACKENDS is not set!"
	exit 255;
fi

# even if some tests do not require it, just warn about this important
# environment variable if it was pre-set
if [[ ! -z ${OMP_NUM_THREADS} ]]; then
	echo "Warning: OMP_NUM_THREADS was set (value was \`${OMP_NUM_THREADS}');"
	echo "         this value may be overwritten during testing."
fi

if [[ ! -d ${INPUT_DIR} ]]; then
	printf "Warning: INPUT_DIR does not exist. "
	printf "Some tests will not run without input datasets.\n"
fi

