#ifndef MULTI_BSP_MODEL_BASELINE_HPP
#define MULTI_BSP_MODEL_BASELINE_HPP

#include <iostream>
#include <sys/types.h>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdint>

#define DEBUG_COST_MODELS
// Utility functions that may be used across different models
namespace cost_models {

/*=====================================================================*/
/*--------------------------HW_model namespace-------------------------*/
/*=====================================================================*/
namespace HW_model
{
    // Hardware parameters structure
    typedef struct HWParameters
    {
        size_t d;       // Number of levels
        // size_t SIMD_size;        // Size of each SIMD operation
        // double r_scalar, r_SIMD;// inverse OPS/s (seconds/op)
        std::vector<double> g;    // inverse BW (seconds/access)
        std::vector<double> ls;   // latency
        std::vector<uint64_t> m;  // available memory NOT on prev levels
        std::vector<size_t> p;    // sub-components
        std::vector<size_t> kmax; // Max number of access streams
    } *HWParameters_p;

    struct HWParameter_configurations
    {
        
      /* A list of the supported thread counts */
    std::vector<uint64_t> threads_options_num;
      /* A list of the names of the supported NUMA allocation policies */
      std::vector<std::string> policy_options_str;
      /* A list of the names of the supported memory levels */
      std::vector<std::string> level_options_str;

      /* The thread ID of the hardware model, e.g. hw_models[i] corresponds to
      / threads_options_num[hw_model_thread_id[i]] threads */
      std::vector<size_t> hw_model_thread_id;

      /* The policy ID of the hardware model, e.g. hw_models[i] corresponds to
      / policy_options_str[hw_model_policy_id[i]] policy */
      std::vector<size_t> hw_model_policy_id;

      /* The level bitmask of the hardware model, e.g. hw_models[i] models the levels from
      / level_options_str that have a 1 in the corresponding bits in hw_model_level_hash[i] */
      std::vector<size_t> hw_model_level_bithash;

      /* A list of the hardware models */
      std::vector < HWParameters > hw_models;
    };

    // Utility function to get thread count for a specific hw_model index
    uint64_t get_threads_for_hw_model(size_t hw_model_idx, HWParameter_configurations config)
    {
        if (hw_model_idx >= config.hw_models.size())
        {
            throw std::invalid_argument("Invalid hw_model index");
        }
        return config.threads_options_num[config.hw_model_thread_id[hw_model_idx]];
    }

    // Utility function to get policy name for a specific hw_model index
    std::string get_policy_for_hw_model(size_t hw_model_idx, HWParameter_configurations config)
    {
        if (hw_model_idx >= config.hw_models.size())
        {
            throw std::invalid_argument("Invalid hw_model index");
        }
        return config.policy_options_str[config.hw_model_policy_id[hw_model_idx]];
    }

    // Utility function to get level names for a specific hw_model index based on bitmask
    std::vector<std::string> get_levels_for_hw_model(size_t hw_model_idx, HWParameter_configurations config)
    {
        if (hw_model_idx >= config.hw_models.size())
        {
            throw std::invalid_argument("Invalid hw_model index");
        }
        
        std::vector<std::string> active_levels;
        size_t bitmask = config.hw_model_level_bithash[hw_model_idx];
        
        for (size_t i = 0; i < config.level_options_str.size(); ++i)
        {
            if (bitmask & (1ULL << i))
            {
                active_levels.push_back(config.level_options_str[i]);
            }
        }
        
        return active_levels;
    }

    // Utility function to find hw_model index by thread count and policy
    size_t find_hw_model_idx(size_t threads, const std::string& policy, HWParameter_configurations config)
    {
        for (size_t i = 0; i < config.hw_models.size(); ++i)
        {
            uint64_t model_threads = config.threads_options_num[config.hw_model_thread_id[i]];
            std::string model_policy = config.policy_options_str[config.hw_model_policy_id[i]];
            
            if (model_threads == threads && model_policy == policy)
            {
                return i;
            }
        }
        
        throw std::invalid_argument("No hw_model found for given threads and policy");
    }

    // Validate HWParameter_configurations structure
    void hw_config_validate(const HWParameter_configurations& config)
    {
        if (config.hw_models.empty())
        {
            throw std::invalid_argument("HWParameter_configurations has no hw_models");
        }
        
        if (config.hw_models.size() != config.hw_model_thread_id.size() ||
            config.hw_models.size() != config.hw_model_policy_id.size() ||
            config.hw_models.size() != config.hw_model_level_bithash.size())
        {
            throw std::invalid_argument("Inconsistent hw_model array sizes in HWParameter_configurations");
        }
        
        // Validate all thread_ids are valid
        for (size_t thread_id : config.hw_model_thread_id)
        {
            if (thread_id >= config.threads_options_num.size())
            {
                throw std::invalid_argument("Invalid thread_id in hw_model_thread_id");
            }
        }
        
        // Validate all policy_ids are valid
        for (size_t policy_id : config.hw_model_policy_id)
        {
            if (policy_id >= config.policy_options_str.size())
            {
                throw std::invalid_argument("Invalid policy_id in hw_model_policy_id");
            }
        }
        
        // Validate all hw_models
        for (const auto& hw_model : config.hw_models)
        {
            // Validate hardware parameters
            if (hw_model.d <= 0 || hw_model.g.empty() ||
                hw_model.ls.empty() || hw_model.m.empty() || hw_model.p.empty() ||
                hw_model.kmax.empty())
            {
                throw std::invalid_argument("Missing required hardware parameters");
            }

            if (hw_model.g.size() != static_cast<size_t>(hw_model.d) ||
                hw_model.ls.size() != static_cast<size_t>(hw_model.d) ||
                hw_model.m.size() != static_cast<size_t>(hw_model.d) ||
                hw_model.p.size() != static_cast<size_t>(hw_model.d) ||
                hw_model.kmax.size() != static_cast<size_t>(hw_model.d))
            {
                throw std::invalid_argument("Inconsistent hardware parameter vector sizes");
            }
        }
    }

    // Validate hardware parameters
    void hw_params_validate(const HWParameters_p hw_params)
    {
        // Validate hardware parameters
        if (hw_params->d <= 0 || hw_params->g.empty() ||
            hw_params->ls.empty() || hw_params->m.empty() || hw_params->p.empty() ||
            hw_params->kmax.empty())
        {
            throw std::invalid_argument("Missing required hardware parameters");
        }

        if (hw_params->g.size() != static_cast<size_t>(hw_params->d) ||
            hw_params->ls.size() != static_cast<size_t>(hw_params->d) ||
            hw_params->m.size() != static_cast<size_t>(hw_params->d) ||
            hw_params->p.size() != static_cast<size_t>(hw_params->d) ||
            hw_params->kmax.size() != static_cast<size_t>(hw_params->d))
        {
            throw std::invalid_argument("Inconsistent hardware parameter vector sizes");
        }
    }

    // Formats bytes into human-readable form (B, KB, MB, GB, etc.)
    inline std::string format_bytes(uint64_t bytes)
    {
        if (bytes == 0)
            return "0 B";
        const char *units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024 && unit < 5)
        {
            size /= 1024;
            unit++;
        }
        std::ostringstream ss;
        if (size < 10)
            ss << std::fixed << std::setprecision(2) << size;
        else if (size < 100)
            ss << std::fixed << std::setprecision(1) << size;
        else
            ss << std::fixed << std::setprecision(0) << size;
        ss << " " << units[unit];
        return ss.str();
    }

    // Print hardware parameters
    void hw_params_print(const HWParameters_p hw_params)
    {
#ifdef DEBUG_COST_MODELS
        std::cout << "\nHardware parameters:\n";
        std::cout << "  - Levels (d): " << hw_params->d << "\n";
        for (size_t i = 0; i < hw_params->d; i++)
        {
            std::cout << "  - Level " << (i + 1) << ":\n";
            std::cout << "    - Bandwidth (1/g): " << std::fixed << std::setprecision(2)
                      << (1.0 / hw_params->g[i] / 1e9) << " GB/s\n";
            std::cout << "    - Latency (ls): " << std::fixed << std::setprecision(2)
                      << (hw_params->ls[i] * 1e9) << " ns\n";
            std::cout << "    - Memory (m): " << format_bytes(hw_params->m[i]) << "\n";
            std::cout << "    - Processing units (p): " << hw_params->p[i] << "\n";
            std::cout << "    - Max streams (kmax): " << hw_params->kmax[i] << "\n";
        }
#endif
    }

    // Print hardware parameters with level names from HWParameter_configurations
    void hw_params_print_with_levels(const HWParameters_p hw_params, 
                                   const std::vector<std::string>& level_names)
    {
#ifdef DEBUG_COST_MODELS
        std::cout << "\nHardware parameters:\n";
        std::cout << "  - Levels (d): " << hw_params->d << "\n";
        for (size_t i = 0; i < hw_params->d; i++)
        {
            std::string level_name = (i < level_names.size()) ? level_names[i] : "Unknown";
            std::cout << "  - Level " << (i + 1) << " (" << level_name << "):\n";
            std::cout << "    - Bandwidth (1/g): " << std::fixed << std::setprecision(2)
                      << (1.0 / hw_params->g[i] / 1e9) << " GB/s\n";
            std::cout << "    - Latency (ls): " << std::fixed << std::setprecision(2)
                      << (hw_params->ls[i] * 1e9) << " ns\n";
            std::cout << "    - Memory (m): " << format_bytes(hw_params->m[i]) << "\n";
            std::cout << "    - Processing units (p): " << hw_params->p[i] << "\n";
            std::cout << "    - Max streams (kmax): " << hw_params->kmax[i] << "\n";
        }
#endif
    }

    /**
     * Select hardware model from HWParameter_configurations
     * 
     * Default strategy:
     * 1. Use policy_id = 0 (default/first policy, typically "close")
     * 2. Use max level_bithash (most comprehensive level coverage)
     * 3. Match thread count to requested threads (exact or closest)
     * 
     * @param num_threads Target thread count
     * @param config HWParameter_configurations structure
     * @return Selected HWParameters
     */
    inline HWParameters select_hw_model_auto(size_t num_threads, 
                                             const HWParameter_configurations& config) {
        
        if (config.hw_models.empty()) {
            throw std::runtime_error("No hardware models available in configuration");
        }
        
        // Step 1: Find all models with policy_id = 0 (default policy)
        std::vector<size_t> policy_0_indices;
        for (size_t i = 0; i < config.hw_models.size(); ++i) {
            if (config.hw_model_policy_id[i] == 0) {
                policy_0_indices.push_back(i);
            }
        }
        
        if (policy_0_indices.empty()) {
            throw std::runtime_error("No hardware models found with policy_id = 0");
        }
        
        // Step 2: Find max level_bithash among policy_0 models
        size_t max_bithash = 0;
        for (size_t idx : policy_0_indices) {
            if (config.hw_model_level_bithash[idx] > max_bithash) {
                max_bithash = config.hw_model_level_bithash[idx];
            }
        }
        
        // Step 3: Filter to models with max level_bithash
        std::vector<size_t> candidates;
        for (size_t idx : policy_0_indices) {
            if (config.hw_model_level_bithash[idx] == max_bithash) {
                candidates.push_back(idx);
            }
        }
        
        // Step 4: Find best thread count match
        size_t best_idx = candidates[0];
        uint64_t best_threads = config.threads_options_num[config.hw_model_thread_id[best_idx]];
        
        for (size_t idx : candidates) {
            uint64_t model_threads = config.threads_options_num[config.hw_model_thread_id[idx]];
            
            // Exact match - use it!
            if (model_threads == num_threads) {
                return config.hw_models[idx];
            }
            
            // Find closest match
            if (std::abs(static_cast<int64_t>(model_threads) - static_cast<int64_t>(num_threads)) <
                std::abs(static_cast<int64_t>(best_threads) - static_cast<int64_t>(num_threads))) {
                best_idx = idx;
                best_threads = model_threads;
            }
        }
        
        return config.hw_models[best_idx];
    }

}
/*=====================================================================*/
/*------------------------hier_roofline namespace----------------------*/
/*=====================================================================*/
    namespace hier_roofline {

        typedef struct AlgoParameters
        {
            uint64_t ops_scalar, ops_SIMD;
            uint64_t b_foot, b_reads, b_writes;
            size_t lvl;
        } *AlgoParameters_p;

        void algo_params_validate(AlgoParameters_p algo_params)
        {
            // Validate input parameters
            if (algo_params == nullptr)
                throw std::invalid_argument("Algorithm parameters cannot be null");
        }

        // Print algorithm parameters
        void algo_params_print(AlgoParameters_p algo_params)
        {
            // Algorithm parameters
#ifdef DEBUG_COST_MODELS
            std::cout << "\nHierarchical-Roofline Model Parameters:\n";

            // Compute operations
            std::cout << "  - Compute Operations:\n";
            std::cout << "    - Scalar operations: " << algo_params->ops_scalar << "\n";
            std::cout << "    - SIMD operations: " << algo_params->ops_SIMD << "\n";
            std::cout << "    - Total operations: " << (algo_params->ops_scalar + algo_params->ops_SIMD) << "\n";

            // Memory operations
            std::cout << "  - Memory Operations:\n";
            std::cout << "    - Memory footprint: " << HW_model::format_bytes(algo_params->b_foot) << "\n";
            std::cout << "    - Read volume: " << HW_model::format_bytes(algo_params->b_reads) << "\n";
            std::cout << "    - Write volume: " << HW_model::format_bytes(algo_params->b_writes) << "\n";
            std::cout << "    - Total data transfer: " << HW_model::format_bytes(algo_params->b_reads + algo_params->b_writes) << "\n";

            // Operational intensity
            double op_intensity = static_cast<double>(algo_params->ops_scalar + algo_params->ops_SIMD) /
                                  (algo_params->b_reads + algo_params->b_writes);
            std::cout << "  - Operational Intensity: " << std::fixed << std::setprecision(2)
                      << op_intensity << " ops/byte\n";

            // Memory level
            if (algo_params->lvl > 0)
            {
                std::cout << "  - Target memory level: " << algo_params->lvl << "\n";
            }
            else
            {
                std::cout << "  - Target memory level: auto\n";
            }
#endif
        }

        // Adjust memory level for supersteps
        void adjust_lvl_naive(HW_model::HWParameters_p hw_params,
                              AlgoParameters_p algo_params, size_t target_threads)
        {
            // The base level for auto-adjustment is the maximum level of all supersteps
            size_t base_level = 0, target_level = hw_params->d;

            if (algo_params->lvl)
                return;

            // Naive search for the appropriate memory level (d)
            size_t pi_mult = 1;
            for (size_t lvl = 0; lvl < hw_params->d; lvl++)
            {
                pi_mult *= hw_params->p[lvl];
                if (lvl >= base_level &&
                    algo_params->b_foot <= hw_params->m[lvl])// && target_threads <= pi_mult)
                {
                    target_level = lvl + 1;
                    break;
                }
            }
            algo_params->lvl = target_level;
        }

/*=====================================================================*/
/*------------------------------Predictor-------------------------------*/
/**
 * Predicts execution cost for a kernel using a Hierarchical Roofline model.
 *
 * @param hw_params Hardware parameters
 * @param algo_params Algorithm parameters
 * @param target_threads Number of threads
 * @return Predicted execution cost in seconds
 */
        double predict_cost(HW_model::HWParameters_p hw_params,
                        AlgoParameters_p algo_params,
                        size_t target_threads){
#ifdef DEBUG_COST_MODELS
            std::cout << "===== Hierarchical-Roofline Cost Prediction =====\n\n";
            std::cout << "Threads: " << target_threads << "\n";
#endif

            algo_params_validate(algo_params);
            algo_params_print(algo_params);
            hw_params_validate(hw_params);
            // hw_params_print(hw_params);

            adjust_lvl_naive(hw_params, algo_params, target_threads);
            double comp_t = 0.0, mem_t = 0.0;
            comp_t = 0; // Currently ignoring compute time
            // (hw_params->r_scalar * algo_params->ops_scalar / target_threads + hw_params->r_SIMD * algo_params->ops_SIMD / target_threads);
            if (algo_params->b_foot)
                mem_t = hw_params->g[algo_params->lvl - 1] 
                * (algo_params->b_reads / target_threads + algo_params->b_writes / target_threads)
                + hw_params->ls[algo_params->lvl - 1];
            else mem_t = 0;

			double total_cost = std::max( comp_t, mem_t );
			std::cout << "\nMemory footprint: " << HW_model::format_bytes( algo_params->b_foot ) << "\n";
			// Print final summary
			std::cout << "\nTotal cost: " << std::scientific << std::setprecision( 4 ) << total_cost << " seconds\n";
			return total_cost;
        }
        /*=====================================================================*/
		/*--------------------------------COO----------------------------------*/
		AlgoParameters_p get_params_coo( uint64_t nz, uint64_t n, uint64_t m, size_t x_dsize, 
            size_t y_dsize, size_t A_dsize, size_t A_rowidx_size, size_t A_colidx_size ) {
			AlgoParameters_p spmv_coo = new AlgoParameters();
			spmv_coo->b_foot = ( A_dsize + A_rowidx_size + A_colidx_size ) * nz + y_dsize * m + x_dsize * n;
			spmv_coo->lvl = 0; // Auto-adjust memory level
			spmv_coo->b_reads = ( A_dsize + A_rowidx_size + A_colidx_size + x_dsize + y_dsize ) * nz;
			spmv_coo->b_writes = y_dsize * nz;
			spmv_coo->ops_scalar = 2 * nz;
            spmv_coo->ops_SIMD = 0;
            return spmv_coo;
		}

		/*=====================================================================*/
        /*--------------------------------CSR----------------------------------*/
		AlgoParameters_p get_params_csr( uint64_t nz, uint64_t n, uint64_t m, size_t y_dsize, size_t x_dsize, size_t A_dsize, size_t A_rowptr_size, size_t A_colidx_size ) {
			AlgoParameters_p spmv_csr = new AlgoParameters();
			spmv_csr->b_foot = ( A_colidx_size + A_dsize ) * nz + A_rowptr_size * ( m + 1 ) + y_dsize * m + x_dsize * n;
			spmv_csr->b_reads = ( A_colidx_size + A_dsize + x_dsize ) * nz + A_rowptr_size * ( m + 1 ) + y_dsize * m;
			spmv_csr->b_writes = y_dsize * m;
			spmv_csr->ops_scalar = 2 * nz;
            spmv_csr->ops_SIMD = 0;
            return spmv_csr;
		}
		/*=====================================================================*/
        /*--------------------------------set----------------------------------*/
		AlgoParameters_p get_params_set( uint64_t n, bool y_vec, size_t x_dsize, size_t y_dsize, bool i ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			if( i ) {
				algo_p->b_foot = 1;
				algo_p->b_reads = 1;
				algo_p->b_writes = 1;
			} else {
				algo_p->b_foot = x_dsize * n + ( y_vec ? y_dsize * n : 0 );
				algo_p->b_reads = 0 + ( y_vec ? y_dsize * n : 0 );
				algo_p->b_writes = x_dsize * n;
			}
			algo_p->ops_scalar = 0;
			algo_p->ops_SIMD = 0;
			algo_p->lvl = 0;
			return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------clear--------------------------------*/
        AlgoParameters_p get_params_clear(uint64_t n, size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = dtype_size * n;
            algo_p->b_reads = 0;
            algo_p->b_writes = dtype_size * n;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 0;
            algo_p->lvl = 0;
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------apply--------------------------------*/
        AlgoParameters_p get_params_apply() //(size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0;
            algo_p->b_reads = 0;
            algo_p->b_writes = 0;
            algo_p->ops_scalar = 1;
            algo_p->ops_SIMD = 0;
            algo_p->lvl = 0;
            return algo_p;
        }
        /*=====================================================================*/
        /*------------------------------eWiseApply-----------------------------*/
		AlgoParameters_p get_params_eWiseApply( uint64_t n, size_t z_dsize, size_t x_dsize, 
            size_t y_dsize, bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = z_dsize * n + ( x_vec ? x_dsize * n : 0 ) + ( y_vec ? y_dsize * n : 0 );
			algo_p->b_reads = ( x_vec ? x_dsize * n : 0 ) + ( y_vec ? y_dsize * n : 0 );
			algo_p->b_writes = z_dsize * n;
			algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------foldl--------------------------------*/
		AlgoParameters_p get_params_foldl( uint64_t n, size_t x_dsize, size_t y_dsize, 
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0 + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_writes = (x_vec ? x_dsize * n : 0);
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------foldr--------------------------------*/
		AlgoParameters_p get_params_foldr( uint64_t n, size_t x_dsize, size_t y_dsize, 
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0 + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_writes = (y_vec ? y_dsize * n : 0);
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------dot---------------------------------*/
		AlgoParameters_p get_params_dot( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = ( x_dsize + y_dsize ) * n;
			algo_p->b_reads = ( x_dsize + y_dsize ) * n;
			algo_p->b_writes = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 2*n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------eWiseAdd---------------------------------*/
		AlgoParameters_p get_params_eWiseAdd( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_writes = z_dsize * n;
			algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------eWiseMul---------------------------------*/
		AlgoParameters_p get_params_eWiseMul( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_writes = z_dsize * n;
			algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------muladd-------------------------------*/
		AlgoParameters_p get_params_muladd( uint64_t n, size_t z_dsize, size_t a_dsize, size_t x_dsize, size_t y_dsize, bool a_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = ( z_dsize + x_dsize + y_dsize ) * n + ( a_vec ? a_dsize * n : 0 );
			algo_p->b_reads = ( x_dsize + y_dsize ) * n + ( a_vec ? a_dsize * n : 0 );
			algo_p->b_writes = (z_dsize)*n;
			algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 2 * n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*==================================================================*/
    }
/*=====================================================================*/
/*----------------------hier_lat_roofline namespace--------------------*/
/*=====================================================================*/
    namespace hier_lat_roofline
    {

        // Algorithm parameters structure
        typedef struct AlgoParameters
        {
            uint64_t ops_scalar, ops_SIMD;
            uint64_t b_foot;
            uint64_t b_reads, b_writes;
            uint64_t rand_reads, rand_writes;

            size_t lvl;
        } *AlgoParameters_p;

        void algo_params_validate(AlgoParameters_p algo_params)
        {
            // Validate input parameters
            if (algo_params == nullptr)
                throw std::invalid_argument("Algorithm parameters cannot be null");
        }

        // Print algorithm parameters
        void algo_params_print(AlgoParameters_p algo_params)
        {
            // Algorithm parameters
#ifdef DEBUG_COST_MODELS
            std::cout << "\nHierarchical-Latency-Aware-Roofline Model Parameters:\n";
            
            // Compute operations
            std::cout << "  - Compute Operations:\n";
            std::cout << "    - Scalar operations: " << algo_params->ops_scalar << "\n";
            std::cout << "    - SIMD operations: " << algo_params->ops_SIMD << "\n";
            std::cout << "    - Total operations: " << (algo_params->ops_scalar + algo_params->ops_SIMD) << "\n";
            
            // Memory operations
            std::cout << "  - Memory Operations:\n";
            std::cout << "    - Memory footprint: " << HW_model::format_bytes(algo_params->b_foot) << "\n";
            
            // Batch operations
            std::cout << "    - Batch reads: " << HW_model::format_bytes(algo_params->b_reads) << "\n";
            std::cout << "    - Batch writes: " << HW_model::format_bytes(algo_params->b_writes) << "\n";
            std::cout << "    - Total batch transfers: " << HW_model::format_bytes(algo_params->b_reads + algo_params->b_writes) << "\n";
            
            // Random operations
            std::cout << "    - Random reads: " << HW_model::format_bytes(algo_params->rand_reads) << "\n";
            std::cout << "    - Random writes: " << HW_model::format_bytes(algo_params->rand_writes) << "\n";
            std::cout << "    - Total random transfers: " << HW_model::format_bytes(algo_params->rand_reads + algo_params->rand_writes) << "\n";
            
            // Total data transfer
            uint64_t total_reads = algo_params->b_reads + algo_params->rand_reads;
            uint64_t total_writes = algo_params->b_writes + algo_params->rand_writes;
            uint64_t total_transfers = total_reads + total_writes;
            
            std::cout << "    - Total reads: " << HW_model::format_bytes(total_reads) << "\n";
            std::cout << "    - Total writes: " << HW_model::format_bytes(total_writes) << "\n";
            std::cout << "    - Total data transfer: " << HW_model::format_bytes(total_transfers) << "\n";
            
            // Operational intensity
            double op_intensity = static_cast<double>(algo_params->ops_scalar + algo_params->ops_SIMD) / 
                                  (total_transfers > 0 ? total_transfers : 1);
            std::cout << "  - Operational Intensity: " << std::fixed << std::setprecision(2)
                      << op_intensity << " ops/byte\n";
            
            // Batch vs Random ratio
            if (total_transfers > 0) {
                double batch_ratio = static_cast<double>(algo_params->b_reads + algo_params->b_writes) / 
                                     total_transfers * 100.0;
                std::cout << "  - Batch access ratio: " << std::fixed << std::setprecision(1)
                          << batch_ratio << "%\n";
            }
            
            // Memory level
            if (algo_params->lvl > 0) {
                std::cout << "  - Target memory level: " << algo_params->lvl << "\n";
            } else {
                std::cout << "  - Target memory level: auto\n";
            }
#endif
        }

        // Adjust memory level for supersteps
        void adjust_lvl_naive(HW_model::HWParameters_p hw_params,
                              AlgoParameters_p algo_params, size_t target_threads)
        {
            // The base level for auto-adjustment is the maximum level of all supersteps
            size_t base_level = 0, target_level = hw_params->d;

            if (algo_params->lvl)
                return;

            // Naive search for the appropriate memory level (d)
            size_t pi_mult = 1;
            for (size_t lvl = 0; lvl < hw_params->d; lvl++)
            {
                pi_mult *= hw_params->p[lvl];
                if (lvl >= base_level &&
                    algo_params->b_foot <= hw_params->m[lvl])// && target_threads <= pi_mult)
                {
                    target_level = lvl + 1;
                    break;
                }
            }
            algo_params->lvl = target_level;
        }

        /*=====================================================================*/
        /*------------------------------Predictor-------------------------------*/
        /**
         * Predicts execution cost for a kernel using a Hierarchical Roofline model with added latency for random access.
         * The assumption is that batch accesses incur only bandwidth cost, while random accesses incur both bandwidth and latency costs.
         *
         * @param hw_params Hardware parameters
         * @param algo_params Algorithm parameters
         * @param target_threads Number of threads
         * @return Predicted execution cost in seconds
         */
        double predict_cost(HW_model::HWParameters_p hw_params,
                            AlgoParameters_p algo_params,
                            size_t target_threads)
        {
#ifdef DEBUG_COST_MODELS
			std::cout << "===== Hierarchical-Latency-Aware-Roofline Cost Prediction =====\n\n";
			std::cout << "Threads: " << target_threads << "\n";
#endif

            algo_params_validate(algo_params);
            algo_params_print(algo_params);
            hw_params_validate(hw_params);
            // hw_params_print(hw_params);

            adjust_lvl_naive(hw_params, algo_params, target_threads);
            double comp_t = 0.0, mem_t = 0.0;
            comp_t = 0; // Currently ignoring compute time
            // (hw_params->r_scalar * algo_params->ops_scalar / target_threads + hw_params->r_SIMD * algo_params->ops_SIMD / target_threads);
            mem_t = hw_params->g[algo_params->lvl - 1] 
            * (algo_params->b_reads / target_threads + algo_params->b_writes / target_threads) 
            + hw_params->ls[algo_params->lvl - 1] 
            * (algo_params->rand_reads / target_threads + algo_params->rand_writes / target_threads);

			double total_cost = std::max(comp_t, mem_t);
            std::cout << "\nMemory footprint: " << HW_model::format_bytes( algo_params->b_foot ) << "\n";
			// Print final summary
			std::cout << "\nTotal cost: " << std::scientific << std::setprecision( 4 ) << total_cost << " seconds\n";
			return total_cost;
        }
        /*=====================================================================*/
        /*--------------------------------COO----------------------------------*/
		AlgoParameters_p get_params_coo( uint64_t nz, uint64_t n, uint64_t m, size_t x_dsize, size_t y_dsize, 
            size_t A_dsize, size_t A_rowidx_size, size_t A_colidx_size ) {
			AlgoParameters_p spmv_coo = new AlgoParameters();
			spmv_coo->b_foot = ( A_dsize + A_rowidx_size + A_colidx_size ) * nz + y_dsize * m + x_dsize * n;
            spmv_coo->lvl = 0; // Auto-adjust memory level
			spmv_coo->b_reads = ( A_dsize + A_rowidx_size + A_colidx_size + x_dsize + y_dsize ) * nz;
			spmv_coo->b_writes = y_dsize * nz;
            spmv_coo->rand_reads = 2 * nz; // * dtype_size
            spmv_coo->rand_writes = nz; // * dtype_size

            return spmv_coo;
		}

		/*=====================================================================*/
        /*--------------------------------CSR----------------------------------*/

		AlgoParameters_p get_params_csr( uint64_t nz, uint64_t n, uint64_t m, size_t y_dsize, 
            size_t x_dsize, size_t A_dsize, size_t A_rowptr_size, size_t A_colidx_size ) {
			AlgoParameters_p spmv_csr = new AlgoParameters();
			spmv_csr->b_foot = ( A_colidx_size + A_dsize ) * nz + A_rowptr_size * ( m + 1 ) + y_dsize * m + x_dsize * n;
			spmv_csr->b_reads = ( A_colidx_size + A_dsize + x_dsize ) * nz + A_rowptr_size * ( m + 1 ) + y_dsize * m;
			spmv_csr->b_writes = y_dsize * m;
			spmv_csr->rand_reads = nz;
            spmv_csr->rand_writes = 0;

            return spmv_csr;
		}
		/*=====================================================================*/
        /*--------------------------------set----------------------------------*/
		AlgoParameters_p get_params_set( uint64_t n, bool y_vec, size_t x_dsize, size_t y_dsize, bool i ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            if (i)
            {
                algo_p->b_foot = 1;
                algo_p->b_reads = 1;
                algo_p->b_writes = 1;
                algo_p->rand_writes = 1;
                algo_p->rand_reads = 1;
            }
            else
            {
				algo_p->b_foot = x_dsize * n + ( y_vec ? y_dsize * n : 0 );
				algo_p->b_reads = 0 + ( y_vec ? y_dsize * n : 0 );
				algo_p->b_writes = x_dsize * n;
				algo_p->rand_writes = 0;
                algo_p->rand_reads = 0;
            }
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 0;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------clear--------------------------------*/
        AlgoParameters_p get_params_clear(uint64_t n, size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = dtype_size * n;
            algo_p->b_reads = 0;
            algo_p->b_writes = dtype_size * n;
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 0;
            algo_p->lvl = 0;
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------apply--------------------------------*/
        AlgoParameters_p get_params_apply() //(size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0;
            algo_p->b_reads = 0;
            algo_p->b_writes = 0;
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 1;
            algo_p->ops_SIMD = 0;
            algo_p->lvl = 0;
            return algo_p;
        }
        /*=====================================================================*/
        /*------------------------------eWiseApply-----------------------------*/
		AlgoParameters_p get_params_eWiseApply( uint64_t n, size_t z_dsize, 
            size_t x_dsize, size_t y_dsize, bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = z_dsize * n +
                             (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_writes = z_dsize * n;
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------foldl--------------------------------*/
		AlgoParameters_p get_params_foldl( uint64_t n, size_t x_dsize, size_t y_dsize, 
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0 + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_writes = (x_vec ? x_dsize * n : 0);
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------foldr--------------------------------*/
		AlgoParameters_p get_params_foldr( uint64_t n, size_t x_dsize, size_t y_dsize, 
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = 0 + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_writes = (y_vec ? y_dsize * n : 0);
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------dot---------------------------------*/
		AlgoParameters_p get_params_dot( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = (x_dsize + y_dsize) * n;
			algo_p->b_reads = (x_dsize + y_dsize) * n;
			algo_p->b_writes = 0;
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 2 * n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------eWiseAdd---------------------------------*/
		AlgoParameters_p get_params_eWiseAdd( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_writes = z_dsize * n;
			algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*---------------------------------eWiseMul---------------------------------*/
		AlgoParameters_p get_params_eWiseMul( uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
            bool x_vec, bool y_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
			algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_reads = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
			algo_p->b_writes = z_dsize * n;
			algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*=====================================================================*/
        /*--------------------------------muladd-------------------------------*/
		AlgoParameters_p get_params_muladd( uint64_t n, size_t z_dsize, size_t a_dsize, 
            size_t x_dsize, size_t y_dsize, bool a_vec ) {
			AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->b_foot = (z_dsize + x_dsize + y_dsize) * n + (a_vec ? a_dsize * n : 0);
            algo_p->b_reads = (x_dsize + y_dsize) * n + (a_vec ? a_dsize * n : 0);
            algo_p->b_writes = (z_dsize) * n;
            algo_p->rand_writes = 0;
            algo_p->rand_reads = 0;
            algo_p->ops_scalar = 0;
            algo_p->ops_SIMD = 2 * n;
            algo_p->lvl = 0;
            return algo_p;
		}
		/*==================================================================*/
    } 

    // k-Multi-BSP performance model
    namespace k_multi_bsp
    {
        // Superstep structure
        typedef struct Superstep
        {
            uint64_t nv; // Number of supersteps of this type
            size_t lvl;  // Memory level
            size_t ops_scalar, ops_SIMD;
            uint64_t ks; // Number of access streams
            // For large ks over same-sized hi, use hi_rep to define after which idx hi values are repeated
            uint64_t hi_rep;
            std::vector<uint64_t> hi; // Access sizes for each stream
        } *Superstep_p;

        // Algorithm parameters structure
        typedef struct AlgoParameters
        {
            uint64_t n;                     // Total number of supersteps
            uint64_t num_v;                 // Number of superstep variations
            uint64_t b_foot;                // Memory footprint  
            std::vector<Superstep_p> ss_v;  // Superstep variations
        } *AlgoParameters_p;

        // Validate algorithm parameters
        void algo_params_validate(AlgoParameters_p algo_params)
        {
            // Validate input parameters
            if (algo_params->num_v <= 0 || algo_params->ss_v.empty())
            {
                throw std::invalid_argument("Missing required algorithm parameters: num_v and ss_v");
            }

            if (algo_params->ss_v.size() != static_cast<size_t>(algo_params->num_v))
            {
                throw std::invalid_argument("ss_v should have " + std::to_string(algo_params->num_v) +
                                            " elements, but got " + std::to_string(algo_params->ss_v.size()));
            }
            // Validate each superstep
            for (size_t i = 0; i < algo_params->ss_v.size(); i++)
            {
                Superstep_p ss = algo_params->ss_v[i];
                if (ss->hi.size() + ss->hi_rep != static_cast<size_t>(ss->ks))
                {
                    throw std::invalid_argument("Superstep type " + std::to_string(i + 1) +
                                                " has ks=" + std::to_string(ss->ks) +
                                                " but hi has " + std::to_string(ss->hi.size()) + " elements");
                }
            }
        }

        // Print algorithm parameters
        void algo_params_print(AlgoParameters_p algo_params)
        {
            // Algorithm parameters
#ifdef DEBUG_COST_MODELS
            std::cout << "\nAlgorithm parameters:\n";
            std::cout << "  - Total supersteps (n): " << algo_params->n << "\n";
            std::cout << "  - Superstep variations (num_v): " << algo_params->num_v << "\n";
            std::cout << "  - Memory footprint: " << HW_model::format_bytes(algo_params->b_foot) << "\n";

            for (size_t i = 0; i < algo_params->ss_v.size(); i++)
            {
                Superstep_p ss = algo_params->ss_v[i];
                uint64_t sum_hi = 0;
                for (uint64_t h : ss->hi)
                {
                    sum_hi += h;
                }

                std::cout << "  - Superstep type " << (i + 1) << ":\n";
                std::cout << "    - Count: " << ss->nv << " ("
                          << std::fixed << std::setprecision(1)
                          << (static_cast<double>(ss->nv) * 100.0 / algo_params->n) << "% of total)\n";
                std::cout << "    - Memory level: " << ss->lvl << "\n";
                std::cout << "    - Access streams (ks): " << ss->ks << "\n";

                std::cout << "    - Access sizes (hi): ";
                for (size_t j = 0; j < ss->hi.size(); j++)
                {
                    std::cout << ss->hi[j];
                    if (j < ss->hi.size() - 1)
                    {
                        std::cout << ", ";
                    }
                    else if (ss->hi_rep > 0)
                    {
                        std::cout << " (X" << ss->hi_rep + 1 << ")";
                    }
                }
                std::cout << "\n";

                std::cout << "    - Volume per superstep: " << HW_model::format_bytes(sum_hi) << "\n";
            }
#endif
        }

        // // Calculate memory footprint
        // uint64_t get_mem_footprint(AlgoParameters_p algo_params)
        // {
        //     uint64_t memory_footprint = 0;

        //     // Calculate total memory footprint across all superstep types
        //     for (size_t i = 0; i < algo_params->ss_v.size(); i++)
        //     {
        //         Superstep_p ss = algo_params->ss_v[i];
        //         uint64_t sum_hi = 0, itter = 0;
        //         for (itter = 0; itter < ss->hi.size(); ++itter)
        //         {
        //             sum_hi += ss->hi[itter];
        //         }
        //         sum_hi += ss->hi_rep * ss->hi[ss->hi.size() - 1];
        //         memory_footprint += static_cast<uint64_t>(ss->nv) * sum_hi;
        //     }
        //     return memory_footprint;
        // }

        // Adjust memory level for supersteps
        void adjust_lvl_naive(HW_model::HWParameters_p hw_params,
                              AlgoParameters_p algo_params, size_t target_threads)
        {
            // The base level for auto-adjustment is the maximum level of all supersteps
            size_t target_level = hw_params->d - 1;
            bool adjust_level = false;
            for (size_t t = 0; t < algo_params->ss_v.size(); t++)
                if (!(algo_params->ss_v[t]->lvl) || algo_params->ss_v[t]->lvl == 42)
                    adjust_level = true;
            if (!adjust_level)
                return;

            // Naive search for the appropriate memory level (d)
            size_t pi_mult = 1;
            for (size_t lvl = 0; lvl < hw_params->d; lvl++)
            {
                pi_mult *= hw_params->p[lvl];
                if (algo_params->b_foot <= hw_params->m[lvl]) // && target_threads <= pi_mult)
                {
                    target_level = lvl + 1;
                    break;
                }
            }
            for (size_t t = 0; t < algo_params->ss_v.size(); t++)
            {
                if (!(algo_params->ss_v[t]->lvl))
                    algo_params->ss_v[t]->lvl = target_level;
                else if (algo_params->ss_v[t]->lvl == 42)
                    algo_params->ss_v[t]->lvl = hw_params->d;
            }
        }

        /*=====================================================================*/
        /*------------------------------Predictor-------------------------------*/
        /**
         * Predicts execution cost for a kernel using the k-Multi-BSP model with configurable stream aggregation.
         *
         * @param hw_params Hardware parameters
         * @param algo_params Algorithm parameters
         * @param target_threads Number of threads
         * @param stream_aggregator Method to aggregate streams ("max" or "sum")
         * @return Predicted execution cost in seconds
         */
        double predict_cost(HW_model::HWParameters_p hw_params,
                            AlgoParameters_p algo_params,
                            size_t target_threads,
                            const std::string &stream_aggregator = "sum")
        {
#ifdef DEBUG_COST_MODELS
            std::cout << "===== k-Multi-BSP Cost Prediction =====\n\n";
            std::cout << "Threads: " << target_threads << "\n";
            std::cout << "Stream aggregator: " << stream_aggregator << "\n";
#endif

            algo_params_validate(algo_params);
            algo_params_print(algo_params);
            hw_params_validate(hw_params);
            // hw_params_print(hw_params);

            adjust_lvl_naive(hw_params, algo_params, target_threads);

            // Calculate cost for each superstep type
            double total_cost = 0.0;
#ifdef DEBUG_COST_MODELS
			std::cout << "\nComputation breakdown by superstep type:\n";
#endif
            for (size_t t = 0; t < algo_params->ss_v.size(); t++)
            {
                Superstep_p ss = algo_params->ss_v[t];
                if (!ss->ks) continue;
                // Extract parameters
                uint64_t num_supersteps = ss->nv;
                size_t lvl = ss->lvl;
                std::vector<uint64_t> &hi = ss->hi;
                uint64_t hi_extras = ss->hi_rep;

                // Ensure level index is valid
                if (lvl <= 0 || lvl > hw_params->d)
                {
                    std::cerr << "Warning: Superstep type " << (t + 1) << " specifies level " << lvl
                              << ", which is out of range. Valid range: 1 to " << hw_params->d << "\n";
                    lvl = std::min(std::max((size_t)1, lvl), hw_params->d);
                }

                // Levels are 1-indexed in the model, adjust for 0-based arrays
                size_t lvl_idx = static_cast<size_t>(lvl - 1);

                double g_level = hw_params->g[lvl_idx];
                double ls_level = hw_params->ls[lvl_idx];
                size_t kmax_level = hw_params->kmax[lvl_idx];

                // Check if ks exceeds kmax
                if (ss->ks > kmax_level)
                {
                    std::cerr << "Warning: Superstep type " << (t + 1) << " has ks=" << ss->ks
                              << " exceeding kmax=" << kmax_level << " for level " << lvl << "\n";
                }

                // Calculate access sizes
                uint64_t sum_hi = 0;
                uint64_t max_hi = 0;

                for (size_t i = 0; i < hi.size(); ++i)
                {
                    sum_hi += hi[i];
                    max_hi = std::max(max_hi, hi[i]);
                }

                // Add repeated elements contribution
                if (hi_extras > 0 && !hi.empty())
                {
                    sum_hi += hi[hi.size() - 1] * hi_extras;
                }

                // Determine which aggregator to use for cost calculation
                uint64_t access_size = (stream_aggregator == "sum") ? sum_hi : max_hi;
                std::string aggregator_name = (stream_aggregator == "sum" ? "sum_hi" : "max_hi");

                // Calculate cost for one superstep: access_size * g + ls
                double superstep_cost = access_size * g_level + ls_level;

                // Total cost for all supersteps of this type
				double type_cost = num_supersteps * superstep_cost;
                total_cost += type_cost;
#ifdef DEBUG_COST_MODELS
				// Print details
                std::cout << "Superstep type " << (t + 1) << " (level " << lvl << "):\n";
                std::cout << "  - Count: " << num_supersteps << "\n";
                std::cout << "  - " << (stream_aggregator == "sum" ? "Sum of" : "Max") << " access size: "
                          << access_size << " bytes\n";
                std::cout << "  - Cost per superstep: "
                          << std::scientific << std::setprecision(4)
                          << superstep_cost << " seconds\n";
                std::cout << "    = " << aggregator_name << "(" << access_size << ") * g(" << g_level << ") + ls(" << ls_level << ")\n";
                std::cout << "  - Total cost: "
                          << std::scientific << std::setprecision(4)
                          << type_cost << " seconds\n";
#endif
			}
#ifdef DEBUG_COST_MODELS

			std::cout << "\nMemory footprint: " << HW_model::format_bytes(algo_params->b_foot) << "\n";
            // Print final summary
            std::cout << "\nTotal cost: "
                      << std::scientific << std::setprecision(4)
                      << total_cost << " seconds\n";
#endif
			return total_cost;
        }

        /*=====================================================================*/
        /*--------------------------------COO----------------------------------*/
        AlgoParameters_p get_params_coo(uint64_t nz, uint64_t n,
                                        uint64_t m, size_t x_dsize, size_t y_dsize,
                                        size_t A_dsize, size_t A_rowidx_size, size_t A_colidx_size)
        {
            AlgoParameters_p spmv_coo = new AlgoParameters();
            spmv_coo->n = nz;
            spmv_coo->b_foot = (A_dsize + A_rowidx_size + A_colidx_size) * nz
                + y_dsize * m + x_dsize * n;
            spmv_coo->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            spmv_coo->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_coo = new Superstep();
            ss_coo->nv = nz;
            ss_coo->ops_scalar = 2;
            ss_coo->ops_SIMD = 0;
            ss_coo->lvl = 0;
            ss_coo->ks = 5;
            ss_coo->hi_rep = 0;
            ss_coo->hi = {A_rowidx_size, A_colidx_size, A_dsize, x_dsize, y_dsize};
            spmv_coo->ss_v.push_back(ss_coo);
            return spmv_coo;
        }

        AlgoParameters_p get_params_coo_batched(uint64_t nz, uint64_t n,
                                                uint64_t m, size_t x_dsize, size_t y_dsize,
                                                size_t A_dsize, size_t A_rowidx_size, size_t A_colidx_size, size_t batch_sz)
        {
            std::cout << "get_params_coo_batched not implemented, falling back to get_params_coo\n";
            (void)batch_sz;
            return get_params_coo(nz, n, m, x_dsize, y_dsize, A_dsize, A_rowidx_size, A_colidx_size);


            // AlgoParameters_p spmv_coo = new AlgoParameters();
            // spmv_coo->n = nz / batch_sz;
            // spmv_coo->num_v = 1;
            // spmv_coo->b_foot = (2 * idx_size + dtype_size) * nz + dtype_size * (m + n);
            // Superstep_p ss_coo = new Superstep();
            // ss_coo->nv = nz / batch_sz;
            // ss_coo->ops_scalar = 2 * batch_sz;
            // ss_coo->ops_SIMD = 0;
            // ss_coo->lvl = 0;
            // ss_coo->ks = 2 * batch_sz + 3;
            // ss_coo->hi_rep = 2 * batch_sz - 1;                                                          // 5 streams
            // ss_coo->hi = {idx_size * batch_sz, idx_size * batch_sz, dtype_size * batch_sz, dtype_size}; // 8 bytes per stream
            // spmv_coo->ss_v.push_back(ss_coo);
            // return spmv_coo;
        }

        /*=====================================================================*/
        /*--------------------------------CSR----------------------------------*/

        AlgoParameters_p get_params_csr(uint64_t nz, uint64_t n,
                                        uint64_t m, size_t y_dsize, size_t x_dsize,
                                        size_t A_dsize, size_t A_rowptr_size, size_t A_colidx_size)
        {
        AlgoParameters_p spmv_csr = new AlgoParameters();
        spmv_csr->n = nz;    // Same number of non-zeros
        spmv_csr->b_foot = (A_colidx_size + A_dsize) * nz + A_rowptr_size * (m + 1)
            + y_dsize * m + x_dsize * n;
        spmv_csr->num_v = 3;
        Superstep_p ss_omp_barrier = new Superstep();
        ss_omp_barrier->nv = 1;
        ss_omp_barrier->ops_scalar = 0;
        ss_omp_barrier->ops_SIMD = 0;
        ss_omp_barrier->lvl = 42;
        ss_omp_barrier->ks = 1;
        ss_omp_barrier->hi_rep = 0;
        ss_omp_barrier->hi = {0};
        spmv_csr->ss_v.push_back(ss_omp_barrier);
        // Superstep A (pipelined) - internal loop
        Superstep_p ss_A = new Superstep();
        ss_A->nv = (nz > m) ? nz - m : 0;
        ss_A->ops_scalar = 2;
        ss_A->ops_SIMD = 0;
        ss_A->lvl = 0;
        ss_A->ks = 3;
        ss_A->hi_rep = 0;
        ss_A->hi = {A_colidx_size, A_dsize, x_dsize};

        // Superstep A + B (pipelined)  - internal loop + row processing + y write
        Superstep_p ss_AB = new Superstep();
        ss_AB->nv = m;
        ss_AB->ops_scalar = 2;
        ss_AB->ops_SIMD = 0;
        ss_AB->lvl = 0;
        ss_AB->ks = 5;
        ss_AB->hi_rep = 0;
        ss_AB->hi = {A_rowptr_size, y_dsize, A_colidx_size, A_dsize, x_dsize};

        spmv_csr->ss_v.push_back(ss_A);
        spmv_csr->ss_v.push_back(ss_AB);
        return spmv_csr;
        }

        AlgoParameters_p get_params_csr_batched(uint64_t nz, uint64_t n,
                                                uint64_t m, size_t x_dsize, size_t y_dsize,
                                                size_t A_dsize, size_t A_rowptr_size, size_t A_colidx_size, size_t batch_sz)
        {
            std::cout << "get_params_csr_batched not implemented, falling back to get_params_csr\n";
            (void)batch_sz;
            return get_params_coo(nz, n, m, x_dsize, y_dsize, A_dsize, A_rowptr_size, A_colidx_size);

            // AlgoParameters_p spmv_csr = new AlgoParameters();
            // spmv_csr->n = nz / batch_sz; // Same number of non-zeros
            // spmv_csr->num_v = 2;           // Two superstep types
            // spmv_csr->b_foot = (idx_size + dtype_size) * nz + dtype_size * (m + n) + idx_size * (m + 1);
            // if (nz < batch_sz * n)
            // {
            //     throw std::invalid_argument("Batch size too large for the given matrix dimensions.");
            // }

            // // Superstep A (pipelined) - internal loop
            // Superstep_p ss_A = new Superstep();
            // ss_A->nv = nz / batch_sz - m;
            // ss_A->ops_scalar = 2 * batch_sz;
            // ss_A->ops_SIMD = 0;
            // ss_A->lvl = 0;
            // ss_A->ks = 2 + batch_sz;
            // ss_A->hi_rep = batch_sz - 1;
            // ss_A->hi = /* A */ {idx_size * batch_sz, dtype_size * batch_sz, dtype_size};

            // // Superstep A + B (pipelined)  - internal loop + row processing + y write
            // Superstep_p ss_AB = new Superstep();
            // ss_AB->nv = m;
            // ss_AB->ops_scalar = 2 * batch_sz;
            // ss_AB->ops_SIMD = 0;
            // ss_AB->lvl = 0;
            // // Assuming that rowPtr and y are also batched/accessed consecutively (works for cachelines...not easy algorithmically)
            // ss_AB->ks = 4 + batch_sz;
            // ss_AB->hi_rep = batch_sz - 1;
            // ss_AB->hi = /* B */ {idx_size, dtype_size,
            //                     /* A */ idx_size * batch_size, dtype_size * batch_size, dtype_size};

            // spmv_csr->ss_v.push_back(ss_A);
            // spmv_csr->ss_v.push_back(ss_AB);
            // return spmv_csr;
        }
        /*=====================================================================*/
        /*--------------------------------set----------------------------------*/
        AlgoParameters_p get_params_set(uint64_t n, bool y_vec, size_t x_dsize,
            size_t y_dsize, bool i)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            if (i)
            {
                algo_p->b_foot = x_dsize;
                ss_A->nv = 1;
                ss_A->ops_scalar = 0;
                ss_A->ops_SIMD = 0;
                ss_A->lvl = 0;
                ss_A->ks = 1;
                ss_A->hi_rep = 0;
                ss_A->hi = {x_dsize};
            }
            else
            {
                algo_p->b_foot = x_dsize * n + (y_vec ? y_dsize * n : 0);
                ss_A->nv = 1;
                ss_A->ops_scalar = 0;
                ss_A->ops_SIMD = 0;
                ss_A->lvl = 0;
                ss_A->ks = 1 + (y_vec ? 1 : 0);
                ss_A->hi_rep = 0;
                if (y_vec)
                    ss_A->hi = {x_dsize * n, y_dsize * n};
                else
                    ss_A->hi = {x_dsize * n};
            }
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------clear--------------------------------*/
        AlgoParameters_p get_params_clear(uint64_t n, size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = dtype_size * n;
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = 0;
            ss_A->lvl = 0;
            ss_A->ks = 1;
            ss_A->hi_rep = 0;
            ss_A->hi = {dtype_size * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------apply--------------------------------*/
        AlgoParameters_p get_params_apply()//(size_t dtype_size)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 1;
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = 0;
            ss_A->nv = 1;
            ss_A->ops_scalar = 1;
            ss_A->ops_SIMD = 0;
            ss_A->lvl = 0;
            ss_A->ks = 0;
            ss_A->hi_rep = 0;
            ss_A->hi = {};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------eWiseApply--------------------------------*/
        AlgoParameters_p get_params_eWiseApply(uint64_t n, size_t z_dsize, size_t x_dsize,
            size_t y_dsize, bool x_vec, bool y_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = z_dsize * n + 
                (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = n;
            ss_A->lvl = 0;
            ss_A->ks = 1 + (x_vec ? 1 : 0) + (y_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (x_vec && y_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n, y_dsize * n};
            else if (x_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n};
            else if (y_vec)
                ss_A->hi = {z_dsize * n, y_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------foldl--------------------------------*/
        AlgoParameters_p get_params_foldl(uint64_t n,
            size_t x_dsize, size_t y_dsize, bool x_vec, bool y_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = (x_vec ?  x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = n;
            ss_A->lvl = 0;
            ss_A->ks = (x_vec ? 1 : 0) + (y_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (x_vec && y_vec)
                ss_A->hi = {x_dsize * n, y_dsize * n};
            else if (x_vec)
                ss_A->hi = {x_dsize * n};
            else if (y_vec)
                ss_A->hi = {y_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------foldr--------------------------------*/
        AlgoParameters_p get_params_foldr(uint64_t n,
            size_t x_dsize, size_t y_dsize, bool x_vec, bool y_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = n;
            ss_A->lvl = 0;
            ss_A->ks = (x_vec ? 1 : 0) + (y_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (x_vec && y_vec)
                ss_A->hi = {x_dsize * n, y_dsize * n};
            else if (x_vec)
                ss_A->hi = {x_dsize * n};
            else if (y_vec)
                ss_A->hi = {y_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*---------------------------------dot---------------------------------*/
        AlgoParameters_p get_params_dot(uint64_t n,
            size_t z_dsize, size_t x_dsize, size_t y_dsize)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = y_dsize * n + x_dsize * n;
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = 2 * n;
            ss_A->lvl = 0;
            ss_A->ks = 2;
            ss_A->hi_rep = 0;
            ss_A->hi = {y_dsize * n, x_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*---------------------------------eWiseAdd---------------------------------*/
        AlgoParameters_p get_params_eWiseAdd(uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
        bool x_vec, bool y_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = n;
            ss_A->lvl = 0;
            ss_A->ks = 1 + (x_vec ? 1 : 0) + (y_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (x_vec && y_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n, y_dsize * n};
            else if (x_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n};
            else if (y_vec)
                ss_A->hi = {z_dsize * n, y_dsize * n};
            else 
                ss_A->hi = {z_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*---------------------------------eWiseMul---------------------------------*/
        AlgoParameters_p get_params_eWiseMul(uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize,
        bool x_vec, bool y_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = z_dsize * n + (x_vec ? x_dsize * n : 0) + (y_vec ? y_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = n;
            ss_A->lvl = 0;
            ss_A->ks = 1 + (x_vec ? 1 : 0) + (y_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (x_vec && y_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n, y_dsize * n};
            else if (x_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n};
            else if (y_vec)
                ss_A->hi = {z_dsize * n, y_dsize * n};
            else 
                ss_A->hi = {z_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*=====================================================================*/
        /*--------------------------------eWiseMuladd-------------------------------*/
        //TODO: NOT UPDATED 
        AlgoParameters_p get_params_eWiseMuladd(uint64_t n,
            size_t z_dsize, size_t a_dsize, size_t x_dsize, size_t y_dsize, bool a_vec)
        {
            AlgoParameters_p algo_p = new AlgoParameters();
            algo_p->n = 1;
            algo_p->num_v = 2;
            Superstep_p ss_omp_barrier = new Superstep();
            ss_omp_barrier->nv = 1;
            ss_omp_barrier->ops_scalar = 0;
            ss_omp_barrier->ops_SIMD = 0;
            ss_omp_barrier->lvl = 42;
            ss_omp_barrier->ks = 1;
            ss_omp_barrier->hi_rep = 0;
            ss_omp_barrier->hi = {0};
            algo_p->ss_v.push_back(ss_omp_barrier);
            Superstep_p ss_A = new Superstep();
            algo_p->b_foot = (z_dsize + x_dsize + y_dsize) * n + (a_vec ? a_dsize * n : 0);
            ss_A->nv = 1;
            ss_A->ops_scalar = 0;
            ss_A->ops_SIMD = 2 * n;
            ss_A->lvl = 0;
            ss_A->ks = 3 + (a_vec ? 1 : 0);
            ss_A->hi_rep = 0;
            if (a_vec)
                ss_A->hi = {z_dsize * n, x_dsize * n, y_dsize * n, a_dsize * n};
            else
                ss_A->hi = {z_dsize * n, x_dsize * n, y_dsize * n};
            algo_p->ss_v.push_back(ss_A);
            return algo_p;
        }
        /*==================================================================*/

    } // namespace multi_bsp
} // namespace cost_models

#endif // MULTI_BSP_MODEL_BASELINE_HPP
