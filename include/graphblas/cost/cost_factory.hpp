#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <typeindex>

#include "hw_params_arm920.hpp"

// Define a compile-time toggle
#ifndef _GRB_ENABLE_TRACING
#define _GRB_ENABLE_TRACING 0  // Default to off
#endif

#ifdef _GRB_ENABLE_TRACING

namespace detail {
    template<size_t... Ints>
    struct index_sequence {
        using type = index_sequence;
        static constexpr size_t size() noexcept { return sizeof...(Ints); }
    };
    
    // Index sequence builder via recursion
    template<size_t N, size_t... Ints>
    struct make_index_sequence_helper : make_index_sequence_helper<N-1, N-1, Ints...> {};
    
    template<size_t... Ints>
    struct make_index_sequence_helper<0, Ints...> {
        using type = index_sequence<Ints...>;
    };
    
    template<size_t N>
    using make_index_sequence = typename make_index_sequence_helper<N>::type;
}


// First, save the original functions before we redefine them
namespace grb {
    namespace original {
        using namespace grb;  // This brings in all the original functions
    }
}

// Forward declarations for the function objects (moved to the top)
struct EWiseApplyFunc;
struct FoldlFunc;
struct FoldrFunc;
struct DotFunc;
struct SetFunc;
struct ApplyFunc;
struct MxvFunc;

// Function to get cost predictor name (moved before its usage)
template<typename Func>
std::string getCostPredictorName() {
    if (std::is_same<Func, EWiseApplyFunc>::value) return "eWiseApply";
    if (std::is_same<Func, FoldlFunc>::value) return "foldl";
    if (std::is_same<Func, FoldrFunc>::value) return "foldr";
    if (std::is_same<Func, DotFunc>::value) return "dot";
    if (std::is_same<Func, SetFunc>::value) return "set";
    if (std::is_same<Func, ApplyFunc>::value) return "apply";
    if (std::is_same<Func, MxvFunc>::value) return "mxv";
    return "unknown";
}

// Type trait to check if we can call grb::size on a type
template<typename T, typename = void>
struct has_grb_size : std::false_type {};

// Specialization for types where grb::size(T) is valid
template<typename T>
struct has_grb_size<T, 
    typename std::enable_if<
        !std::is_same<
            decltype(grb::size(std::declval<T>())),
            void
        >::value
    >::type
> : std::true_type {};

// Helper to safely get size if available
template<typename T>
typename std::enable_if<has_grb_size<T>::value, std::string>::type
getSizeString(const T& arg) {
    try {
        return "[size=" + std::to_string(grb::size(arg)) + "] ";
    } catch(...) {
        return " ";
    }
}

// Helper for types that don't support size
template<typename T>
typename std::enable_if<!has_grb_size<T>::value, std::string>::type
getSizeString(const T&) {
    return " ";
}

// Add these type traits to detect Matrix types safely
template<typename T, typename = void>
struct has_grb_matrix_functions : std::false_type {};

// Specialization for types where grb::nnz(T), grb::nrows(T), and grb::ncols(T) are valid
template<typename T>
struct has_grb_matrix_functions<T, 
    typename std::enable_if<
        !std::is_same<
            decltype(grb::nnz(std::declval<T>())),
            void
        >::value &&
        !std::is_same<
            decltype(grb::nrows(std::declval<T>())),
            void
        >::value &&
        !std::is_same<
            decltype(grb::ncols(std::declval<T>())),
            void
        >::value
    >::type
> : std::true_type {};

// Helper to get matrix dimensions and nnz if available
template<typename T>
typename std::enable_if<has_grb_matrix_functions<T>::value, std::string>::type
getMatrixInfoString(const T& arg) {
    try {
        return "[rows=" + std::to_string(grb::nrows(arg)) + 
               ",cols=" + std::to_string(grb::ncols(arg)) +
               ",nnz=" + std::to_string(grb::nnz(arg)) + "] ";
    } catch(...) {
        return " ";
    }
}

// Helper for types that don't support matrix functions
template<typename T>
typename std::enable_if<!has_grb_matrix_functions<T>::value, std::string>::type
getMatrixInfoString(const T&) {
    return " ";
}

// Primary template for operator name traits - default case
template<typename T>
struct OperatorNameTrait {
    static std::string name() { return typeid(T).name(); }
};

// Specializations for common GraphBLAS operators
// Basic arithmetic operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::add<Args...>> {
    static std::string name() { return "operators::add"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::mul<Args...>> {
    static std::string name() { return "operators::mul"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::subtract<Args...>> {
    static std::string name() { return "operators::subtract"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::divide<Args...>> {
    static std::string name() { return "operators::divide"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::divide_reverse<Args...>> {
    static std::string name() { return "operators::divide_reverse"; }
};

// Min/Max operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::min<Args...>> {
    static std::string name() { return "operators::min"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::max<Args...>> {
    static std::string name() { return "operators::max"; }
};

// Logical operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::logical_or<Args...>> {
    static std::string name() { return "operators::logical_or"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::logical_and<Args...>> {
    static std::string name() { return "operators::logical_and"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::any_or<Args...>> {
    static std::string name() { return "operators::any_or"; }
};

// Comparison operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::equal<Args...>> {
    static std::string name() { return "operators::equal"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::not_equal<Args...>> {
    static std::string name() { return "operators::not_equal"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::less_than<Args...>> {
    static std::string name() { return "operators::less_than"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::greater_than<Args...>> {
    static std::string name() { return "operators::greater_than"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::leq<Args...>> {
    static std::string name() { return "operators::leq"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::geq<Args...>> {
    static std::string name() { return "operators::geq"; }
};

// Other common operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::abs_diff<Args...>> {
    static std::string name() { return "operators::abs_diff"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::square_diff<Args...>> {
    static std::string name() { return "operators::square_diff"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::relu<Args...>> {
    static std::string name() { return "operators::relu"; }
};

// Assignment operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::left_assign<Args...>> {
    static std::string name() { return "operators::left_assign"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::right_assign<Args...>> {
    static std::string name() { return "operators::right_assign"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::left_assign_if<Args...>> {
    static std::string name() { return "operators::left_assign_if"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::right_assign_if<Args...>> {
    static std::string name() { return "operators::right_assign_if"; }
};

// Special purpose operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::argmin<Args...>> {
    static std::string name() { return "operators::argmin"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::argmax<Args...>> {
    static std::string name() { return "operators::argmax"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::zip<Args...>> {
    static std::string name() { return "operators::zip"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::equal_first<Args...>> {
    static std::string name() { return "operators::equal_first"; }
};

// Complex operators
template<typename... Args> 
struct OperatorNameTrait<grb::operators::conjugate_mul<Args...>> {
    static std::string name() { return "operators::conjugate_mul"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::conjugate_left_mul<Args...>> {
    static std::string name() { return "operators::conjugate_left_mul"; }
};

template<typename... Args> 
struct OperatorNameTrait<grb::operators::conjugate_right_mul<Args...>> {
    static std::string name() { return "operators::conjugate_right_mul"; }
};

// Template to check if type is a GraphBLAS operator
template<typename T>
struct is_graphblas_operator {
private:
    template<typename U>
    static auto test(int) -> decltype(
        std::declval<U>().template getAdditiveOperator<void>(), 
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value || grb::is_operator<T>::value;
};

// Helper to get type names
template<typename T>
std::string getTypeName() {
    std::string type_name = typeid(T).name();
    
    // Simple demangling for common types
    if (std::is_same<T, double>::value) return "double";
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, int>::value) return "int";
    if (std::is_same<T, unsigned int>::value) return "unsigned int";
    if (std::is_same<T, long>::value) return "long";
    if (std::is_same<T, unsigned long>::value) return "unsigned long";
    if (std::is_same<T, size_t>::value) return "size_t";
    if (std::is_same<T, char>::value) return "char";
    if (std::is_same<T, bool>::value) return "bool";
    
    // GraphBLAS Vector type detection - explicit common cases
    if (std::is_same<T, grb::Vector<double>>::value) return "Vector<double>";
    if (std::is_same<T, grb::Vector<float>>::value) return "Vector<float>";
    if (std::is_same<T, grb::Vector<int>>::value) return "Vector<int>";
    if (std::is_same<T, grb::Vector<unsigned int>>::value) return "Vector<unsigned int>";
    if (std::is_same<T, grb::Vector<long>>::value) return "Vector<long>";
    if (std::is_same<T, grb::Vector<unsigned long>>::value) return "Vector<unsigned long>";
    
    // GraphBLAS Matrix type detection - expanded for more types
    if (std::is_same<T, grb::Matrix<double>>::value) return "Matrix<double>";
    if (std::is_same<T, grb::Matrix<float>>::value) return "Matrix<float>";
    if (std::is_same<T, grb::Matrix<int>>::value) return "Matrix<int>";
    if (std::is_same<T, grb::Matrix<unsigned int>>::value) return "Matrix<unsigned int>";
    if (std::is_same<T, grb::Matrix<long>>::value) return "Matrix<long>";
    if (std::is_same<T, grb::Matrix<unsigned long>>::value) return "Matrix<unsigned long>";
    if (std::is_same<T, grb::Matrix<char>>::value) return "Matrix<char>";
    if (std::is_same<T, grb::Matrix<bool>>::value) return "Matrix<bool>";
    
    // Use operator traits for all operators
    if (grb::is_operator<T>::value) {
        return OperatorNameTrait<T>::name() + "<...>";
    }
    
    // Check for semiring
    if (grb::is_semiring<T>::value) {
        return "Semiring<...>";
    }
    
    // Better fallback mechanism - extract type name from mangled name
    if (type_name.find("Vector") != std::string::npos) return "Vector<...>";
    if (type_name.find("Matrix") != std::string::npos) return "Matrix<...>";
    
    // Improved operator detection in mangled names
    if (type_name.find("operators") != std::string::npos) {
        // Try to extract the operator name
        const std::vector<std::pair<std::string, std::string>> op_names = {
            {"add", "operators::add<...>"},
            {"mul", "operators::mul<...>"},
            {"subtract", "operators::subtract<...>"},
            {"divide", "operators::divide<...>"},
            {"min", "operators::min<...>"},
            {"max", "operators::max<...>"},
            {"identity", "operators::identity"},
            {"logical_or", "operators::logical_or"},
            {"logical_and", "operators::logical_and"},
            {"any_or", "operators::any_or"},
            {"equal", "operators::equal<...>"},
            {"not_equal", "operators::not_equal<...>"},
            {"less_than", "operators::less_than<...>"},
            {"greater_than", "operators::greater_than<...>"},
            {"leq", "operators::leq<...>"},
            {"geq", "operators::geq<...>"},
            {"abs_diff", "operators::abs_diff<...>"},
            {"square_diff", "operators::square_diff<...>"},
            {"relu", "operators::relu<...>"},
            {"argmin", "operators::argmin<...>"},
            {"argmax", "operators::argmax<...>"},
            {"left_assign", "operators::left_assign<...>"},
            {"right_assign", "operators::right_assign<...>"},
            {"left_assign_if", "operators::left_assign_if<...>"},
            {"right_assign_if", "operators::right_assign_if<...>"}
        };
        
        for (const auto& op : op_names) {
            if (type_name.find(op.first) != std::string::npos) {
                return op.second;
            }
        }
        
        // Generic fallback for operators
        return "operators::...";
    }
    
    return type_name;
}

// Forward declaration for printArgTypes
template<typename... Args>
void printArgTypes(Args&&... args);

// Base case
void printArgTypesHelper() {
    // End of recursion
}

// Recursive case
template<typename T, typename... Args>
void printArgTypesHelper(T&& arg, Args&&... args) {
    // Get the type name
    std::string type_name = getTypeName<typename std::decay<T>::type>();
    
    // Print the type name
    std::cout << type_name;
    
    // If it's a Matrix type, print matrix dimensions and nnz
    if (type_name.find("Matrix<") != std::string::npos) {
        std::cout << getMatrixInfoString<typename std::remove_reference<T>::type>(arg);
    }
    // Otherwise if it's a Vector type, print its size
    else if (type_name.find("Vector<") != std::string::npos) {
        std::cout << getSizeString<typename std::remove_reference<T>::type>(arg);
    }
    // For other types, just print a space
    else {
        std::cout << " ";
    }
    
    // Continue with remaining arguments
    printArgTypesHelper(std::forward<Args>(args)...);
}

// Entry point for printing argument types
template<typename... Args>
void printArgTypes(Args&&... args) {
    std::cout << "[TRACING] Argument types: ";
    printArgTypesHelper(std::forward<Args>(args)...);
    std::cout << std::endl;
}

// Cost prediction framework
// Base template for cost prediction
template<typename Func, typename... Args>
struct CostPredictor {
    // Helper to get argument type names for diagnostic purposes
    template<typename T>
    static std::string getArgTypeName() {
        return getTypeName<T>();
    }
    
    // Helper to build a comma-separated list of argument type names
    template<size_t... Is>
    static std::string getArgTypeNamesHelper(detail::index_sequence<Is...>) {
        std::string result;
        // Use fold expression in C++17, but for C++11 we need this workaround
        using expander = int[];
        (void)expander{0, (void(
            result += (Is == 0 ? "" : ", ") + getArgTypeName<typename std::tuple_element<Is, std::tuple<Args...>>::type>()
        ), 0)...};
        return result;
    }
    
    static std::string getArgTypeNames() {
        return getArgTypeNamesHelper(detail::make_index_sequence<sizeof...(Args)>{});
    }
    
    // Rest of the implementation remains the same
    static double predict(const Args&... args) {
        // Silence unused parameter warnings with a fold-expression-like trick
        int unused[] = { 0, (void(args), 0)... };
        (void)unused;  // Silence unused variable warning

        // Enhanced diagnostic message with function name and argument types
        std::string funcName = getCostPredictorName<Func>();
        std::string argTypes = getArgTypeNames();
        
        std::cout << "[WARNING] *** MISSING COST MODEL ***" << std::endl;
        std::cout << "[WARNING] No specialized cost model for: " << funcName << std::endl;
        std::cout << "[WARNING] With argument types: " << argTypes << std::endl;
        std::cout << "[WARNING] To fix this, add a specialization like:" << std::endl;
        std::cout << "[WARNING] template<...appropriate template params...>" << std::endl;
        std::cout << "[WARNING] struct CostPredictor<" << funcName << "Func, " << argTypes << "> {" << std::endl;
        std::cout << "[WARNING]     static double predict(...) { ... }" << std::endl;
        std::cout << "[WARNING] };" << std::endl;
        
        return 1.0; // Default cost
    }
};

// Special case for the void template parameters - needed for SFINAE detection
template<>
struct CostPredictor<void, void> {
    static double predict() {
        // Always fail with a clear message
        // TODO:: enable assertions in the final code 
        // static_assert(!std::is_same<void, void>::value, 
        //     "Non-implemented cost function detected");
        return 1.0;
    }
};
// Type trait to detect if a specialized cost predictor exists
template<typename Func, typename... Args>
struct has_specialized_cost_predictor {
private:
    // Test function - returns true_type if specialized, false_type if base template
    template<typename F, typename... A>
    static constexpr auto test(int) 
        -> decltype(
            CostPredictor<F, A...>::predict(std::declval<A>()...),
            std::integral_constant<bool, 
                !std::is_same<
                    decltype(&CostPredictor<F, A...>::predict),
                    decltype(&CostPredictor<void, void>::predict)
                >::value
            >()
        );
    
    // Fallback function
    template<typename F, typename... A>
    static constexpr std::false_type test(...);

public:
    // Result of the test
    static constexpr bool value = decltype(test<Func, Args...>(0))::value;
};

// Function object wrappers for each GraphBLAS function
struct EWiseApplyFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::eWiseApply(std::forward<Args>(args)...)) {
        return grb::original::eWiseApply(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::eWiseApply<descr>(std::forward<Args>(args)...)) {
        return grb::original::eWiseApply<descr>(std::forward<Args>(args)...);
    }
};

struct FoldlFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::foldl(std::forward<Args>(args)...)) {
        return grb::original::foldl(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::foldl<descr>(std::forward<Args>(args)...)) {
        return grb::original::foldl<descr>(std::forward<Args>(args)...);
    }
};

struct FoldrFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::foldr(std::forward<Args>(args)...)) {
        return grb::original::foldr(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::foldr<descr>(std::forward<Args>(args)...)) {
        return grb::original::foldr<descr>(std::forward<Args>(args)...);
    }
};

struct DotFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::dot(std::forward<Args>(args)...)) {
        return grb::original::dot(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::dot<descr>(std::forward<Args>(args)...)) {
        return grb::original::dot<descr>(std::forward<Args>(args)...);
    }
};

struct SetFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::set(std::forward<Args>(args)...)) {
        return grb::original::set(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::set<descr>(std::forward<Args>(args)...)) {
        return grb::original::set<descr>(std::forward<Args>(args)...);
    }
};

struct ApplyFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::apply(std::forward<Args>(args)...)) {
        return grb::original::apply(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::apply<descr>(std::forward<Args>(args)...)) {
        return grb::original::apply<descr>(std::forward<Args>(args)...);
    }
};

struct MxvFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(grb::original::mxv(std::forward<Args>(args)...)) {
        return grb::original::mxv(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(grb::original::mxv<descr>(std::forward<Args>(args)...)) {
        return grb::original::mxv<descr>(std::forward<Args>(args)...);
    }
};

// Specializations of CostPredictor for different function/argument combinations

/*=====================================================================*/
/*--------------------------------foldl--------------------------------*/
template<typename T1, typename Monoid>
struct CostPredictor<FoldlFunc, grb::Vector<T1>, grb::Vector<T1>, Monoid> {
    static double predict(grb::Vector<T1>& v1, const grb::Vector<T1>& v2, const Monoid&) {
        try {
			size_t n = grb::size( v1 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl( n, size_data , 1, 1);
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(...) {
            return 1.0; // Fallback value
        }
    }
};

template< typename T1, typename Monoid >
struct CostPredictor< FoldlFunc, T1, grb::Vector< T1 >, Monoid > {
	static double predict( T1 & v1, const grb::Vector< T1 > & v2, const Monoid & ) {
		try {
			size_t n = grb::size( v2 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl( n, size_data, 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

template< typename T1, typename Monoid >
struct CostPredictor< FoldlFunc, grb::Vector< T1 >, T1, Monoid > {
	static double predict(grb::Vector< T1 > & v1, const T1 & v2, const Monoid & ) {
		try {
			size_t n = grb::size( v1 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl( n, size_data, 1, 0 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

/*=====================================================================*/
/*--------------------------------foldr--------------------------------*/
template< typename T1, typename Monoid >
struct CostPredictor< FoldrFunc, grb::Vector< T1 >, grb::Vector< T1 >, Monoid > {
	static double predict( grb::Vector< T1 > & v1, const grb::Vector< T1 > & v2, const Monoid & ) {
		try {
			size_t n = grb::size( v1 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, size_data, 1, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

template< typename T1, typename Monoid >
struct CostPredictor< FoldrFunc, T1, grb::Vector< T1 >, Monoid > {
	static double predict( T1 & v1, const grb::Vector< T1 > & v2, const Monoid & ) {
		try {
			size_t n = grb::size( v2 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, size_data, 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

template< typename T1, typename Monoid >
struct CostPredictor< FoldrFunc, grb::Vector< T1 >, T1, Monoid > {
	static double predict( grb::Vector< T1 > & v1, const T1 & v2, const Monoid & ) {
		try {
			size_t n = grb::size( v1 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, size_data, 1, 0 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

// Specializations for dot product
// Catch-all specialization for dot with exactly 5 arguments of any type
template<typename T0, typename VecType, typename MonoidType, typename OpType>
struct CostPredictor<DotFunc, T0, grb::Vector<VecType>, grb::Vector<VecType>, MonoidType, OpType> {
    static double predict(T0 result, grb::Vector<VecType> v1, grb::Vector<VecType> v2, MonoidType monoid, OpType op) {
        // Extract type information for diagnostics
        std::string t1_name = getTypeName<grb::Vector<VecType>>();
        std::string t2_name = getTypeName<grb::Vector<VecType>>();
        std::string t3_name = getTypeName<MonoidType>();
        std::string t4_name = getTypeName<OpType>();
        
        std::cout << "[TRACING] Arg types: " << getTypeName<T0>() << ", " 
                  << t1_name << ", " << t2_name << ", " 
                  << t3_name << ", " << t4_name << std::endl;
        
        try {
            // Try to get the size of the vectors
            size_t n = 0;
            if (t1_name.find("Vector") != std::string::npos) {
                try { n = grb::size(v1); } catch(...) {}
            }
            
            if (n == 0 && t2_name.find("Vector") != std::string::npos) {
                try { n = grb::size(v2); } catch(...) {}
            }
            
            if (n == 0) {
                return 1.0; // Fallback if size can't be determined
            }
            
            // Check for conjugate operations
            bool is_conjugate = t4_name.find("conjugate") != std::string::npos;
            
            // Use appropriate cost model
            cost_models::HW_model::HWParameters hw_model = 
                cost_models::HW_model::get_hw_params_for_threads(1, dis_system_params);
            cost_models::k_multi_bsp::AlgoParameters_p algo_model = 
                cost_models::k_multi_bsp::get_params_dot(n, sizeof(double));
            
            double base_cost = cost_models::k_multi_bsp::predict_cost(&hw_model, algo_model, 1);
            
            // Additional cost for conjugate operations
            double multiplier = is_conjugate ? 1.0 : 1.0;
            return base_cost * multiplier;
            
        } catch(...) {
            return 1.0; // Fallback value
        }
    }
};

/*=====================================================================*/
/*----------------------------------mxv--------------------------------*/
// Specialization for mxv with Semiring
template<typename T, typename SRingType>
struct CostPredictor<MxvFunc, grb::Vector<T>, grb::Matrix<T>, grb::Vector<T>, SRingType> {
    static double predict(const grb::Vector<T>& y, const grb::Matrix<T>& A, const grb::Vector<T>& x, const SRingType& ring) {
        try {
            size_t nnz = grb::nnz(A);
            size_t m = grb::nrows(A);
            size_t n = grb::ncols(A);
            size_t size_idx = sizeof(size_t);
            size_t size_data = sizeof(T);
            // TODO: Implement cost model prediction
            return 1.0;
        } catch(...) {
            return 1.0; // Fallback value
        }
    }
};

// Generic specialization for mxv with any semiring type
template<typename VecType, typename MatType, typename SRType>
struct CostPredictor<MxvFunc, VecType, MatType, VecType, SRType> {
    static double predict(const VecType& y, const MatType& A, const VecType& x, const SRType& ring) {
        // Check if we're dealing with appropriate types
        std::string y_name = getTypeName<VecType>();
        std::string A_name = getTypeName<MatType>();
        std::string x_name = getTypeName<VecType>();
        std::string ring_name = getTypeName<SRType>();
        
        std::cout << "[TRACING] Using generic mxv predictor with types: " 
                  << y_name << ", " << A_name << ", " << x_name << ", " << ring_name << std::endl;
        
        // Only apply cost model if we're working with a Matrix and two Vectors
        if (A_name.find("Matrix") != std::string::npos && 
            y_name.find("Vector") != std::string::npos && 
            x_name.find("Vector") != std::string::npos) {
            
            try {
                size_t nnz = grb::nnz(A);
                size_t m = grb::nrows(A);
                size_t n = grb::ncols(A);
                size_t size_idx = sizeof(size_t);
                size_t size_data = sizeof(double); // Assume double as fallback
                
            // TODO: Implement cost model prediction
            return 1.0;
            } catch(...) {
                // Fall through to default
            }
        }
        
        // Default fallback cost
        return 1.0;
    }
};



// Specialization for eWiseApply with two vectors
template< typename T1, typename T2, typename Op >
struct CostPredictor< EWiseApplyFunc, grb::Vector< T1 >, grb::Vector< T2 >, Op > {
	static double predict( const grb::Vector< T1 > & v1, const grb::Vector< T2 > & v2, const Op & ) {
		try {
			size_t n = grb::size( v1 ), size_data = sizeof( T1 );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_eWiseApply( n, size_data, 1, 0);
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch( ... ) {
			return 1.0; // Fallback value
		}
	}
};

// Specialization for eWiseApply with three vectors and an operator
template<typename T, typename Op>
struct CostPredictor<EWiseApplyFunc, grb::Vector<T>, grb::Vector<T>, grb::Vector<T>, Op> {
    static double predict(const grb::Vector<T>& v1, const grb::Vector<T>& v2, const grb::Vector<T>& v3, const Op&) {
        try {
			size_t n = grb::size( v1 ), size_data = sizeof(T);
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_eWiseApply( n, size_data, 1, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(...) {
            return 1.0; // Fallback value
        }
    }
};

/*=====================================================================*/
/*----------------------------------set--------------------------------*/
// Specialization for set with Vector<double> to Vector<double>
template<>
struct CostPredictor<SetFunc, grb::Vector<double>, grb::Vector<double>> {
    static double predict(const grb::Vector<double>& dst, const grb::Vector<double>& src) {
        try {
            // TODO: Implement proper cost model for vector-to-vector set operation
            std::cout << "[TRACING] Using specialized set(Vector<double>, Vector<double>) predictor" << std::endl;
            size_t n = grb::size(dst);
            // return dummy cost
            // TODO: Implement proper cost model for vector-to-vector set operation
            return 1.0;
        } catch(...) {
            return 1.0; // Fallback value
        }
    }
};

// Generic specialization for set with any Vector<T> to Vector<T>
template<typename T>
struct CostPredictor<SetFunc, grb::Vector<T>, grb::Vector<T>> {
    static double predict(const grb::Vector<T>& dst, const grb::Vector<T>& src) {
        try {
            // TODO: Implement proper cost model for vector-to-vector set operation
            std::cout << "[TRACING] Using specialized set(Vector<T>, Vector<T>) predictor" << std::endl;
            size_t n = grb::size(dst);
            // return dummy cost
            // TODO: Implement proper cost model for vector-to-vector set operation
            return 1.0;
        } catch(...) {
            return 1.0; // Fallback value
        }
    }
};

// Specialization for set with Vector<T> to scalar
template<typename T>
struct CostPredictor<SetFunc, grb::Vector<T>, T> {
    static double predict(const grb::Vector<T>& dst, const T& scalar) {
        try {
            // TODO: Implement proper cost model for vector-to-scalar set operation
            std::cout << "[TRACING] Using specialized set(Vector<T>, scalar) predictor" << std::endl;
            size_t n = grb::size(dst);
            // return dummy cost
            // TODO: Implement proper cost model for vector-to-scalar set operation
            return 1.0;
        } catch(...) {
            return 1.0; // Fallback value
        }
    }
};

// Function tracer class template for handling tracing logic
template<typename Func>
class FunctionTracer {
public:
    FunctionTracer(const std::string& name) : name_(name) {}
    
    // Version for non-templated calls
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(std::declval<Func>()(std::forward<Args>(args)...)) {
        std::cout << "\n[TRACING] Entering function: " << name_ << " with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        // Predict the cost
        double predicted_cost = CostPredictor<Func, typename std::decay<Args>::type...>::predict(args...);
        
        // Check if we used a specialized predictor
        bool has_specialized = has_specialized_cost_predictor<Func, typename std::decay<Args>::type...>::value;
        
        std::cout << "[TRACING] Predicted cost: " << predicted_cost 
                  << " units (cost model: " << getCostPredictorName<Func>();
        
        if (!has_specialized) {
            std::cout << " - DEFAULT MODEL";
        }
        
        std::cout << ")" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        Func func;
        auto result = func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << name_ << " (took " 
                  << duration.count() << "μs)" << std::endl;
        
        // Calculate and report cost/time ratio
        double cost_time_ratio = predicted_cost / static_cast<double>(duration.count());
        std::cout << "[TRACING] Cost/time ratio: " << cost_time_ratio 
                  << " cost units per microsecond" << std::endl;
        
        return result;
    }
    
    // Version for templated calls with descriptor
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(std::declval<Func>().template withDescriptor<descr>(std::forward<Args>(args)...)) {
        std::string descriptor_name = std::to_string(descr);
        if (descr == grb::descriptors::dense) descriptor_name = "dense";
        if (descr == grb::descriptors::structural) descriptor_name = "structural";
        
        std::cout << "\n[TRACING] Entering function: " << name_ << "<" << descriptor_name << "> with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        // Predict the cost
        double predicted_cost = CostPredictor<Func, typename std::decay<Args>::type...>::predict(args...);
        
        // Check if we used a specialized predictor
        bool has_specialized = has_specialized_cost_predictor<Func, typename std::decay<Args>::type...>::value;
        
        std::cout << "[TRACING] Predicted cost: " << predicted_cost 
                  << " units (cost model: " << getCostPredictorName<Func>();

        if (!has_specialized) {
            std::cout << " - DEFAULT MODEL";
        }
        
        std::cout << ")" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        Func func;
        auto result = func.template withDescriptor<descr>(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << name_ << "<" << descriptor_name << "> (took " 
                  << duration.count() << "μs)" << std::endl;
        
        // Calculate and report cost/time ratio
        double cost_time_ratio = predicted_cost / static_cast<double>(duration.count());
        std::cout << "[TRACING] Cost/time ratio: " << cost_time_ratio 
                  << " cost units per microsecond" << std::endl;
        
        return result;
    }
    
private:
    std::string name_;
};

// Now redefine the functions in the grb namespace with tracing
namespace grb {
    // Create tracers for each function
    static const FunctionTracer<EWiseApplyFunc> eWiseApplyTracer("eWiseApply");
    static const FunctionTracer<FoldlFunc> foldlTracer("foldl");
    static const FunctionTracer<FoldrFunc> foldrTracer("foldr");
    static const FunctionTracer<DotFunc> dotTracer("dot");
    static const FunctionTracer<SetFunc> setTracer("set");
    static const FunctionTracer<ApplyFunc> applyTracer("apply");
    static const FunctionTracer<MxvFunc> mxvTracer("mxv");
    
    // Non-templated versions
    template<typename... Args>
    auto eWiseApply(Args&&... args)
        -> decltype(original::eWiseApply(std::forward<Args>(args)...)) {
        return eWiseApplyTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto foldl(Args&&... args)
        -> decltype(original::foldl(std::forward<Args>(args)...)) {
        return foldlTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto foldr(Args&&... args)
        -> decltype(original::foldr(std::forward<Args>(args)...)) {
        return foldrTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto dot(Args&&... args)
        -> decltype(original::dot(std::forward<Args>(args)...)) {
        return dotTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto set(Args&&... args)
        -> decltype(original::set(std::forward<Args>(args)...)) {
        return setTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto apply(Args&&... args)
        -> decltype(original::apply(std::forward<Args>(args)...)) {
        return applyTracer(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    auto mxv(Args&&... args)
        -> decltype(original::mxv(std::forward<Args>(args)...)) {
        return mxvTracer(std::forward<Args>(args)...);
    }
    
    // Templated versions with descriptor
    template<unsigned int descr, typename... Args>
    auto eWiseApply(Args&&... args)
        -> decltype(original::eWiseApply<descr>(std::forward<Args>(args)...)) {
        return eWiseApplyTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto foldl(Args&&... args)
        -> decltype(original::foldl<descr>(std::forward<Args>(args)...)) {
        return foldlTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto foldr(Args&&... args)
        -> decltype(original::foldr<descr>(std::forward<Args>(args)...)) {
        return foldrTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto dot(Args&&... args)
        -> decltype(original::dot<descr>(std::forward<Args>(args)...)) {
        return dotTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto set(Args&&... args)
        -> decltype(original::set<descr>(std::forward<Args>(args)...)) {
        return setTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto apply(Args&&... args)
        -> decltype(original::apply<descr>(std::forward<Args>(args)...)) {
        return applyTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto mxv(Args&&... args)
        -> decltype(original::mxv<descr>(std::forward<Args>(args)...)) {
        return mxvTracer.template withDescriptor<descr>(std::forward<Args>(args)...);
    }
}

#endif // _GRB_ENABLE_TRACING