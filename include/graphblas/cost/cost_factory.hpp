#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <vector>
#include <type_traits>

// Define a compile-time toggle
#ifndef _GRB_ENABLE_TRACING
#define _GRB_ENABLE_TRACING 0  // Default to off
#endif

#if _GRB_ENABLE_TRACING

// First, save the original functions before we redefine them
namespace grb {
    namespace original {
        using namespace grb;  // This brings in all the original functions
    }
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

// Helper for printing argument types
template<typename... Args>
void printArgTypes(Args&&... args);

// Base case
void printArgTypesHelper() {
    // End of recursion
}

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
    
    // GraphBLAS Operator detection
    if (std::is_same<T, grb::operators::add<double>>::value) return "operators::add<double>";
    if (std::is_same<T, grb::operators::add<float>>::value) return "operators::add<float>";
    if (std::is_same<T, grb::operators::add<int>>::value) return "operators::add<int>";
    if (std::is_same<T, grb::operators::mul<double>>::value) return "operators::mul<double>";
    if (std::is_same<T, grb::operators::mul<float>>::value) return "operators::mul<float>";
    if (std::is_same<T, grb::operators::mul<int>>::value) return "operators::mul<int>";
    
    // Generic fallbacks
    if (type_name.find("Vector") != std::string::npos) return "Vector<...>";
    if (type_name.find("Matrix") != std::string::npos) return "Matrix<...>";
    if (type_name.find("operators::") != std::string::npos) return "operators::...";
    
    return type_name;
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

// Function tracer class template for handling tracing logic
template<typename Func>
class FunctionTracer {
public:
    FunctionTracer(const std::string& name) : name_(name) {}
    
    // Version for non-templated calls
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(std::declval<Func>()(std::forward<Args>(args)...)) {
        std::cout << "[TRACING] Entering function: " << name_ << " with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        Func func;
        auto result = func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << name_ << " (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Version for templated calls with descriptor
    template<unsigned int descr, typename... Args>
    auto withDescriptor(Args&&... args) const
        -> decltype(std::declval<Func>().template withDescriptor<descr>(std::forward<Args>(args)...)) {
        std::string descriptor_name = std::to_string(descr);
        if (descr == grb::descriptors::dense) descriptor_name = "dense";
        if (descr == grb::descriptors::structural) descriptor_name = "structural";
        
        std::cout << "[TRACING] Entering function: " << name_ << "<" << descriptor_name << "> with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        Func func;
        auto result = func.template withDescriptor<descr>(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << name_ << "<" << descriptor_name << "> (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
private:
    std::string name_;
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