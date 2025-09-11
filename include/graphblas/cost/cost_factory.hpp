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

// Add this near the top of the file, after other #defines
#ifndef _GRB_COST_MODEL_TEST_MODE
#define _GRB_COST_MODEL_TEST_MODE 0  // Default to off
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

    template<typename T>
    struct is_internal_lambda {
        static constexpr bool value = false;
    };
    
    // // Detect the specific lambda types used in fold operations
    // template<unsigned int descr, bool left, bool sparse, bool masked, bool monoid, 
    //          typename MaskType, typename IOType, typename IType, typename OP, typename Coords>
    // struct is_internal_lambda<grb::internal::fold_from_vector_to_vector_generic<descr, left, sparse, masked, monoid, 
    //                              MaskType, IOType, IType, OP, Coords>> {
    //     static constexpr bool value = true;
    // };

}


// First, save the original functions before we redefine them
namespace grb {
    namespace original {
        // Don't use "using namespace grb" as it creates ambiguity
        using ::grb::eWiseApply;
        using ::grb::foldl;
        using ::grb::foldr;
        //using ::grb::dot;
        using ::grb::set;
        using ::grb::apply;
        using ::grb::mxv;
        using ::grb::eWiseAdd;
        using ::grb::vxm;
        using ::grb::eWiseLambda;
        //typedef ::grb::eWiseLambda eWiseLambda_original;
        using ::grb::mxm;
        using ::grb::zip;
        using ::grb::outer;
        using ::grb::select;
        using ::grb::clear;
    }
}

// Forward declarations for the function objects
struct EWiseApplyFunc;
struct FoldlFunc;
struct FoldrFunc;
struct DotFunc;
struct SetFunc;
struct ApplyFunc;
struct MxvFunc;
struct EWiseAddFunc; 
struct VxmFunc;
struct EWiseLambdaFunc;
struct MxmFunc;
struct ZipFunc;
struct OuterFunc;
struct SelectFunc;
struct ClearFunc;

// Type trait to check at compile time if a function has a corresponding tracer
template<typename Func>
struct has_tracer {
    static constexpr bool value = false;
};

// Specializations for each supported function
template<> struct has_tracer<EWiseApplyFunc> { static constexpr bool value = true; };
template<> struct has_tracer<FoldlFunc> { static constexpr bool value = true; };
template<> struct has_tracer<FoldrFunc> { static constexpr bool value = true; };
template<> struct has_tracer<DotFunc> { static constexpr bool value = true; };
template<> struct has_tracer<SetFunc> { static constexpr bool value = true; };
template<> struct has_tracer<ApplyFunc> { static constexpr bool value = true; };
template<> struct has_tracer<MxvFunc> { static constexpr bool value = true; };
template<> struct has_tracer<EWiseAddFunc> { static constexpr bool value = true; };
template<> struct has_tracer<VxmFunc> { static constexpr bool value = true; };
template<> struct has_tracer<EWiseLambdaFunc> { static constexpr bool value = true; };
template<> struct has_tracer<MxmFunc> { static constexpr bool value = true; };
template<> struct has_tracer<ZipFunc> { static constexpr bool value = true; };
template<> struct has_tracer<OuterFunc> { static constexpr bool value = true; };
template<> struct has_tracer<SelectFunc> { static constexpr bool value = true; };
template<> struct has_tracer<ClearFunc> { static constexpr bool value = true; };

// Primary template for function name trait
template<typename Func>
struct FunctionNameTrait {
    static constexpr const char* name = "unknown";
};

// Specializations for each function type
template<> struct FunctionNameTrait<EWiseApplyFunc> { static constexpr const char* name = "eWiseApply"; };
template<> struct FunctionNameTrait<FoldlFunc> { static constexpr const char* name = "foldl"; };
template<> struct FunctionNameTrait<FoldrFunc> { static constexpr const char* name = "foldr"; };
template<> struct FunctionNameTrait<DotFunc> { static constexpr const char* name = "dot"; };
template<> struct FunctionNameTrait<SetFunc> { static constexpr const char* name = "set"; };
template<> struct FunctionNameTrait<ApplyFunc> { static constexpr const char* name = "apply"; };
template<> struct FunctionNameTrait<MxvFunc> { static constexpr const char* name = "mxv"; };
template<> struct FunctionNameTrait<EWiseAddFunc> { static constexpr const char* name = "eWiseAdd"; };
template<> struct FunctionNameTrait<VxmFunc> { static constexpr const char* name = "vxm"; };
template<> struct FunctionNameTrait<EWiseLambdaFunc> { static constexpr const char* name = "eWiseLambda"; };
template<> struct FunctionNameTrait<MxmFunc> { static constexpr const char* name = "mxm"; };
template<> struct FunctionNameTrait<ZipFunc> { static constexpr const char* name = "zip"; };
template<> struct FunctionNameTrait<OuterFunc> { static constexpr const char* name = "outer"; };
template<> struct FunctionNameTrait<SelectFunc> { static constexpr const char* name = "select"; };
template<> struct FunctionNameTrait<ClearFunc> { static constexpr const char* name = "clear"; };

// Simple function to get cost predictor name using the trait
template<typename Func>
constexpr const char* getCostPredictorName() {
    return FunctionNameTrait<Func>::name;
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

// Primary template for operator name traits - delegates to existing traits when possible
template<typename T>
struct OperatorNameTrait {
    // For operators that have a defined grb::operator_name trait
    template<typename U = T>
    static auto name_impl(int) -> 
        decltype(std::string(grb::operator_name<U>::name)) {
        return std::string(grb::operator_name<U>::name);
    }
    
    // Fallback for types without an grb::operator_name trait
    template<typename U = T>
    static std::string name_impl(...) {
        return typeid(U).name();
    }
    
    static std::string name() {
        return name_impl<T>(0);
    }
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
    if (std::is_same<T, grb::Vector<bool>>::value) return "Vector<bool>";

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

        // error if _GRB_COST_MODEL_TEST_MODE is enabled
        if (_GRB_COST_MODEL_TEST_MODE) {
            std::cerr << "[ERROR] Missing cost model for this function." << std::endl;
            // print function name
            std::cerr << "[ERROR] Function name: " << funcName << std::endl;
        }
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
        std::cerr << "[ERROR] Missing cost model for this function." << std::endl;
        // print function name
        std::cerr << "[ERROR] Function name: " << getCostPredictorName<void>() << std::endl;
        throw std::runtime_error("Missing cost model");
        // return 1.0;
    }
};


// Type tags
struct specialized_cost_predictor_tag {};
struct default_cost_predictor_tag {};

// Completely pure template implementation
template<typename Func, typename... Args>
struct has_specialized_cost_predictor {
private:
    // Helper trait to check if types are the same
    template<typename T1, typename T2>
    struct is_not_same : std::true_type {};
    
    template<typename T>
    struct is_not_same<T, T> : std::false_type {};
    
    // Check if specialization exists by comparing to base template
    static constexpr bool has_specialization = 
        is_not_same<CostPredictor<Func, Args...>, CostPredictor<void, void>>::value;
    
public:
    // Pure template conditional logic
    using type = typename std::conditional<
        has_specialization,
        specialized_cost_predictor_tag,
        default_cost_predictor_tag
    >::type;
};

// Helper alias for cleaner usage
template<typename Func, typename... Args>
using cost_predictor_category = typename has_specialized_cost_predictor<Func, Args...>::type;

// Function object wrappers for each GraphBLAS function
struct EWiseApplyFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::eWiseApply(std::forward<Args>(args)...)) {
        return ::grb::eWiseApply(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::eWiseApply<descr>(std::forward<Args>(args)...)) {
        return ::grb::eWiseApply<descr>(std::forward<Args>(args)...);
    }
};

struct FoldlFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::foldl(std::forward<Args>(args)...)) {
        return ::grb::original::foldl(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::foldl<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::foldl<descr>(std::forward<Args>(args)...);
    }
};

struct FoldrFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::foldr(std::forward<Args>(args)...)) {
        return ::grb::original::foldr(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::foldr<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::foldr<descr>(std::forward<Args>(args)...);
    }
};

struct DotFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::dot(std::forward<Args>(args)...)) {
        return ::grb::dot(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::dot<descr>(std::forward<Args>(args)...)) {
        return ::grb::dot<descr>(std::forward<Args>(args)...);
    }
};

struct SetFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::set(std::forward<Args>(args)...)) {
        return ::grb::original::set(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::set<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::set<descr>(std::forward<Args>(args)...);
    }
};

struct ApplyFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::apply(std::forward<Args>(args)...)) {
        return ::grb::original::apply(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::apply<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::apply<descr>(std::forward<Args>(args)...);
    }
};

struct MxvFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::mxv(std::forward<Args>(args)...)) {
        return ::grb::original::mxv(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::mxv<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::mxv<descr>(std::forward<Args>(args)...);
    }
};

struct EWiseAddFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::eWiseAdd(std::forward<Args>(args)...)) {
        return ::grb::original::eWiseAdd(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::eWiseAdd<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::eWiseAdd<descr>(std::forward<Args>(args)...);
    }
};

struct VxmFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> grb::RC {
        return ::grb::original::vxm(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::vxm<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::vxm<descr>(std::forward<Args>(args)...);
    }
};

struct EWiseLambdaFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::eWiseLambda(std::forward<Args>(args)...)) {
        return ::grb::original::eWiseLambda(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::eWiseLambda<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::eWiseLambda<descr>(std::forward<Args>(args)...);
    }
};

struct MxmFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::mxm(std::forward<Args>(args)...)) {
        return ::grb::original::mxm(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::mxm<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::mxm<descr>(std::forward<Args>(args)...);
    }
};

struct ZipFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::zip(std::forward<Args>(args)...)) {
        return ::grb::original::zip(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::zip<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::zip<descr>(std::forward<Args>(args)...);
    }
};

struct OuterFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::outer(std::forward<Args>(args)...)) {
        return ::grb::original::outer(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::outer<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::outer<descr>(std::forward<Args>(args)...);
    }
};

struct SelectFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::select(std::forward<Args>(args)...)) {
        return ::grb::original::select(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::select<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::select<descr>(std::forward<Args>(args)...);
    }
};

struct ClearFunc {
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(::grb::original::clear(std::forward<Args>(args)...)) {
        return ::grb::original::clear(std::forward<Args>(args)...);
    }
    
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(::grb::original::clear<descr>(std::forward<Args>(args)...)) {
        return ::grb::original::clear<descr>(std::forward<Args>(args)...);
    }
};

// Specializations of CostPredictor for different function/argument combinations
/*=====================================================================*/
/*--------------------------------mxv--------------------------------*/

template< typename T1, typename T2, typename T3, grb::Backend Backend, typename RowIndexType, typename ColIndexType, typename NonzeroIndexType, typename SRingType >
struct CostPredictor< MxvFunc, grb::Vector< T1 >, grb::Matrix< T2, Backend, RowIndexType, ColIndexType, NonzeroIndexType >, grb::Vector< T3 >, SRingType > {
    static double predict( const grb::Vector< T1 > & y, const grb::Matrix< T2, Backend, RowIndexType, ColIndexType, NonzeroIndexType > & A, const grb::Vector< T3 > & x, const SRingType & ring ) {
        try {
            size_t nnz = grb::nnz( A ), m = grb::size( y ), n = grb::size( x );
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
            cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_csr(
                nnz, n, m, sizeof( T1 ), sizeof( T3 ), sizeof( T2 ), sizeof( NonzeroIndexType ), sizeof( RowIndexType ) );
            return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<MxvFunc>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<MxvFunc> with Matrix");
        }
    }
};

/*=====================================================================*/
/*--------------------------------set--------------------------------*/

template< typename T1, typename T2 >
struct CostPredictor< SetFunc, grb::Vector< T1 >, grb::Vector< T2 > > {
    static double predict( grb::Vector< T1 > & x, grb::Vector< T2 > & y ){
        try {
            size_t n = grb::size( x );
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );

            cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_set( n, 1, sizeof( T1 ), sizeof( T2 ), 0 );
            return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<SetFunc, Vector, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<SetFunc> with Vector to Vector");
        }
    }
};

template< typename T1 >
struct CostPredictor< SetFunc, grb::Vector< T1 >, T1 > { 
    static double predict( grb::Vector< T1 > & x, T1 & y ){
        try {
			size_t n = grb::size( x );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );

            cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_set( n, 0, sizeof( T1 ), sizeof( T1 ), 0 );
            return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<SetFunc, Vector, scalar>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<SetFunc> with Vector to scalar");
        }
    }
};

/*=====================================================================*/
/*--------------------------------clear--------------------------------*/

/*=====================================================================*/
/*--------------------------------apply--------------------------------*/
template< typename T1, typename T2, typename T3, typename Op >
struct CostPredictor< ApplyFunc, T1, T2, T3, Op > {
    static double predict( T1 & x, T2 y, T3 z, const Op &op ){
        try {
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
            cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_apply();
            return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<ApplyFunc>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<ApplyFunc>");
        }
    }
};

/*=====================================================================*/
/*------------------------------eWiseApply-----------------------------*/

// Specialization for eWiseApply with two vectors and y scalar
template< typename T1, typename T2, typename T3, typename Op >
struct CostPredictor< EWiseApplyFunc, grb::Vector< T1 >, grb::Vector< T2 >, T3, Op > {
	static double predict( const grb::Vector< T1 > & z, const grb::Vector< T2 > & x, T3 & y, const Op & ) {
		try {
			size_t n = grb::size( z );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_eWiseApply( n, 
                sizeof( T1 ), sizeof( T2 ), sizeof( T3 ), 1, 0 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<EWiseApplyFunc, Vector, Vector, scalar>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<EWiseApplyFunc> with Vector, Vector, scalar");
        }
	}
};

// Specialization for eWiseApply with two vectors and x scalar
template< typename T1, typename T2, typename T3, typename Op >
struct CostPredictor< EWiseApplyFunc, grb::Vector< T1 >, T2, grb::Vector< T3 >, Op > {
	static double predict( const grb::Vector< T1 > & z, T2 & x, const grb::Vector< T3 > & y, const Op & ) {
		try {
			size_t n = grb::size( z );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_eWiseApply
                ( n, sizeof( T1 ), sizeof( T2 ), sizeof( T3 ), 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<EWiseApplyFunc, Vector, scalar, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<EWiseApplyFunc> with Vector, scalar, Vector");
        }
	}
};

// Specialization for eWiseApply with three vectors and an operator
template< typename T1, typename T2, typename T3, typename Op >
struct CostPredictor< EWiseApplyFunc, grb::Vector< T1 >, grb::Vector< T2 > , grb::Vector< T3 >, Op > {
	static double predict( const grb::Vector< T1 > & z, const grb::Vector< T2 > & x, const grb::Vector< T3 > & y, const Op & ) {
		try {
			size_t n = grb::size( z );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_eWiseApply
                ( n, sizeof( T1 ), sizeof( T2 ), sizeof( T3 ), 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<EWiseApplyFunc, Vector, Vector, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<EWiseApplyFunc> with three Vectors");
        }
	}
};

/*=====================================================================*/
/*--------------------------------foldl--------------------------------*/
// ( uint64_t n, size_t x_dsize, size_t y_dsize, bool x_vec, bool y_vec )

template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldlFunc, grb::Vector< T1 >, grb::Vector< T2 >, Monoid > {
	static double predict(grb::Vector<T1>& x, const grb::Vector<T2>& y, const Monoid&) {
        try {
            size_t n = grb::size( x );
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl
                ( n, sizeof( T1 ), sizeof( T2 ), 1, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldlFunc, Vector, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldlFunc> with Vector, Vector");
        }
    }
};

template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldlFunc, T1 , grb::Vector< T2 >, Monoid > {
	static double predict( T1 & x, const grb::Vector< T2 > & y, const Monoid & ) {
        try {
            size_t n = grb::size( y );
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl
                ( n, sizeof( T1 ), sizeof( T2 ), 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldlFunc, scalar, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldlFunc> with scalar, Vector");
        }
    }
};

template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldlFunc, grb::Vector< T1 >, T2 , Monoid > {
	static double predict(grb::Vector< T1 > & x, const T2 & y, const Monoid & ) {
        try {
            size_t n = grb::size( x );
            cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldl
                ( n, sizeof( T1 ), sizeof( T2 ), 1, 0 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldlFunc, Vector, scalar>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldlFunc> with Vector, scalar");
        }
    }
};

/*=====================================================================*/
/*--------------------------------foldr--------------------------------*/
template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldrFunc, grb::Vector< T1 > , grb::Vector< T2 > , Monoid > {
	static double predict(const grb::Vector< T1 > & x, grb::Vector< T2 > & y, const Monoid & ) {
		try {
			size_t n = grb::size( y );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, sizeof( T1 ), sizeof( T2 ), 1, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldrFunc, Vector, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldrFunc> with Vector, Vector");
        }
	}
};

template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldrFunc, T1, grb::Vector< T2 >, Monoid > {
	static double predict( const T1 & x, grb::Vector< T2 > & y, const Monoid & ) {
		try {
			size_t n = grb::size( y );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, sizeof( T1 ), sizeof( T2 ), 0, 1 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldrFunc, scalar, Vector>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldrFunc> with scalar, Vector");
        }
	}
};

template< typename T1, typename T2, typename Monoid >
struct CostPredictor< FoldrFunc, grb::Vector< T1 >, T2, Monoid > {
	static double predict(const grb::Vector< T1 > & x, T2 & y, const Monoid & ) {
		try {
			size_t n = grb::size( x );
			cost_models::HW_model::HWParameters hw_model = cost_models::HW_model::get_hw_params_for_threads( 1, dis_system_params );
			cost_models::k_multi_bsp::AlgoParameters_p algo_model = cost_models::k_multi_bsp::get_params_foldr( n, sizeof( T1 ), sizeof( T2 ), 1, 0 );
			return cost_models::k_multi_bsp::predict_cost( &hw_model, algo_model, 1 );
		} catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<FoldrFunc, Vector, scalar>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<FoldrFunc> with Vector, scalar");
        }
	}
};

/*=====================================================================*/
/*--------------------------------dot----------------------------------*/
// (uint64_t n, size_t z_dsize, size_t x_dsize, size_t y_dsize)
// Catch-all specialization for dot with exactly 5 arguments of any type
template< typename T0, typename T1, typename T2, typename MonoidType, typename OpType >
struct CostPredictor< DotFunc, T0, T1, T2, MonoidType, OpType > {
	static double predict(T0 z, T1 x, T2 y, MonoidType monoid, OpType op) {
        std::cout << "[TRACING] Using catch-all 5-argument dot predictor" << std::endl;

        // Extract type information for diagnostics
        std::string t1_name = getTypeName<T1>();
        std::string t2_name = getTypeName<T2>();
        std::string t3_name = getTypeName<MonoidType>();
        std::string t4_name = getTypeName<OpType>();

        std::cout << "[TRACING] Arg types: " << getTypeName<T0>() << ", "
                  << t1_name << ", " << t2_name << ", "
                  << t3_name << ", " << t4_name << std::endl;

        try {
            // Try to get the size of the vectors
            size_t n = 0;
            if (t1_name.find("Vector") != std::string::npos) {
                try { n = grb::size(x); } catch(...) {}
            }

            if (n == 0 && t2_name.find("Vector") != std::string::npos) {
                try { n = grb::size(y); } catch(...) {}
            }

            if (n == 0) {
                throw std::runtime_error("Could not determine vector size");
            }

            // Check for conjugate operations
            bool is_conjugate = t4_name.find("conjugate") != std::string::npos;

            // Use appropriate cost model
            cost_models::HW_model::HWParameters hw_model =
                cost_models::HW_model::get_hw_params_for_threads(1, dis_system_params);
            cost_models::k_multi_bsp::AlgoParameters_p algo_model =
                cost_models::k_multi_bsp::get_params_dot(n, sizeof(T0), sizeof(T1), sizeof(T2));

            double base_cost = cost_models::k_multi_bsp::predict_cost(&hw_model, algo_model, 1);

            // Additional cost for conjugate operations
            double multiplier = is_conjugate ? 1.0 : 1.0;
            return base_cost * multiplier;

        } catch(const std::exception& e) {
            throw std::runtime_error("Error in CostPredictor<DotFunc>: " + std::string(e.what()));
        } catch(...) {
            throw std::runtime_error("Unknown error in CostPredictor<DotFunc> with 5 arguments");
        }
    }
};

/*=====================================================================*/
/*---------------------------------add---------------------------------*/

/*=====================================================================*/
/*---------------------------------mul---------------------------------*/

/*=====================================================================*/
/*--------------------------------muladd-------------------------------*/

// Function tracer class template for handling tracing logic
template< typename Func >
class FunctionTracer {
private:
    std::string name_;
    
    // Compile-time helper to print model type - specialized version
    template<typename CategoryTag, typename... Args>
    struct ModelTypePrinter {
        static void print() {
            // Default: do nothing (for specialized models)
        }
    };
    
    // Specialization for default models
    template<typename... Args>
    struct ModelTypePrinter<default_cost_predictor_tag, Args...> {
        static void print() {
            std::cout << " - DEFAULT MODEL";
        }
    };

public:
    FunctionTracer(const std::string& name) : name_(name) {
        static_assert(has_tracer<Func>::value, 
            "Missing tracer implementation for a GraphBLAS function. "
            "Please add appropriate entries in cost_factory.hpp for this function.");
    }

    // Single unified version that handles both cases with default template argument
    template<unsigned int descr = 0, typename... Args>
    grb::RC operator()(Args&&... args) const {
        // Build function name with descriptor info if provided
        std::string function_name = name_;
        if (descr != 0) {
            std::string descriptor_name = std::to_string(descr);
            if (descr == grb::descriptors::dense)
                descriptor_name = "dense";
            if (descr == grb::descriptors::structural)
                descriptor_name = "structural";
            function_name += "<" + descriptor_name + ">";
        }
        
        std::cout << "\n[TRACING] Entering function: " << function_name 
                  << " with " << sizeof...(args) << " arguments" << std::endl;

        printArgTypes(std::forward<Args>(args)...);

        double predicted_cost = 0.0;
        
        try {
            predicted_cost = CostPredictor<Func, typename std::decay<Args>::type...>::predict(args...);
            
            std::cout << "[TRACING] Predicted cost: " << predicted_cost 
                      << " units (cost model: " << getCostPredictorName<Func>();

            // Compile-time dispatch - zero runtime overhead
            using predictor_category = cost_predictor_category<Func, typename std::decay<Args>::type...>;
            ModelTypePrinter<predictor_category, Args...>::print();

            std::cout << ")" << std::endl;
        } catch(const std::exception& e) {
            std::cout << "[ERROR] Cost prediction failed: " << e.what() << std::endl;
            
            #if _GRB_COST_MODEL_TEST_MODE
                return grb::FAILED;
            #else
                throw;
            #endif
        } catch(...) {
            std::cout << "[ERROR] Cost prediction failed with unknown exception" << std::endl;
            
            #if _GRB_COST_MODEL_TEST_MODE
                return grb::FAILED;
            #else
                throw;
            #endif
        }

        auto start = std::chrono::high_resolution_clock::now();
        Func func;
        
        // Call the appropriate function based on whether descriptor is provided
        grb::RC result;
        if (descr == 0) {
            result = func(std::forward<Args>(args)...);
        } else {
            result = func.template withDescriptor<descr>(std::forward<Args>(args)...);
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << function_name 
                  << " (took " << duration.count() << "μs)" << std::endl;

        double cost_time_ratio = predicted_cost / static_cast<double>(duration.count());
        std::cout << "[TRACING] Cost/time ratio: " << cost_time_ratio 
                  << " cost units per microsecond" << std::endl;

        return result;
    }
    
    // Convenience method for explicit descriptor calls
    template<unsigned int descr, typename... Args>
    grb::RC withDescriptor(Args&&... args) const {
        return operator()<descr>(std::forward<Args>(args)...);
    }
};


        // Now redefine the functions in the grb namespace with tracing
		namespace grb {
			// Create tracers for each function
			static const FunctionTracer< EWiseApplyFunc > eWiseApplyTracer( "eWiseApply" );
			static const FunctionTracer< FoldlFunc > foldlTracer( "foldl" );
			static const FunctionTracer< FoldrFunc > foldrTracer( "foldr" );
			static const FunctionTracer< DotFunc > dotTracer( "dot" );
			static const FunctionTracer< SetFunc > setTracer( "set" );
			static const FunctionTracer< ApplyFunc > applyTracer( "apply" );
			static const FunctionTracer< MxvFunc > mxvTracer( "mxv" );
            static const FunctionTracer< EWiseAddFunc > eWiseAddTracer( "eWiseAdd" );
            static const FunctionTracer< VxmFunc > vxmTracer( "vxm" );
            static const FunctionTracer< EWiseLambdaFunc > eWiseLambdaTracer( "eWiseLambda" );
            static const FunctionTracer< MxmFunc > mxmTracer( "mxm" );
            static const FunctionTracer< ZipFunc > zipTracer( "zip" );
            static const FunctionTracer< OuterFunc > outerTracer( "outer" );
            static const FunctionTracer< SelectFunc > selectTracer( "select" );
            static const FunctionTracer< ClearFunc > clearTracer( "clear" );


            // Non-templated versions (descriptor = 0 by default)
            template<typename... Args>
            grb::RC eWiseApply(Args&&... args) {
                return eWiseApplyTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC foldl(Args&&... args) {
                return foldlTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC foldr(Args&&... args) {
                return foldrTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC dot(Args&&... args) {
                return dotTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC set(Args&&... args) {
                return setTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC apply(Args&&... args) {
                return applyTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC mxv(Args&&... args) {
                return mxvTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC eWiseAdd(Args&&... args) {
                return eWiseAddTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC vxm(Args&&... args) {
                return vxmTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC eWiseLambda(Args&&... args) {
                return eWiseLambdaTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC mxm(Args&&... args) {
                return mxmTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC zip(Args&&... args) {
                return zipTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC outer(Args&&... args) {
                return outerTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC select(Args&&... args) {
                return selectTracer(std::forward<Args>(args)...);
            }

            template<typename... Args>
            grb::RC clear(Args&&... args) {
                return clearTracer(std::forward<Args>(args)...);
            }

            // Templated versions with explicit descriptor
            template<unsigned int descr, typename... Args>
            grb::RC eWiseApply(Args&&... args) {
                return eWiseApplyTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC foldl(Args&&... args) {
                return foldlTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC foldr(Args&&... args) {
                return foldrTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC dot(Args&&... args) {
                return dotTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC set(Args&&... args) {
                return setTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC apply(Args&&... args) {
                return applyTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC mxv(Args&&... args) {
                return mxvTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC eWiseAdd(Args&&... args) {
                return eWiseAddTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            // TODO:: Implement eWiseLambda with descriptor
            // template<unsigned int descr, typename... Args>
            // grb::RC eWiseLambda(Args&&... args) {
            //     return eWiseLambdaTracer.template operator()<descr>(std::forward<Args>(args)...);
            // }

            template<unsigned int descr, typename... Args>
            grb::RC vxm(Args&&... args) {
                return vxmTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC mxm(Args&&... args) {
                return mxmTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC zip(Args&&... args) {
                return zipTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC outer(Args&&... args) {
                return outerTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC select(Args&&... args) {
                return selectTracer.template operator()<descr>(std::forward<Args>(args)...);
            }

            template<unsigned int descr, typename... Args>
            grb::RC clear(Args&&... args) {
                return clearTracer.template operator()<descr>(std::forward<Args>(args)...);
            }



        } // namespace grb

#endif // _GRB_ENABLE_TRACING