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
        // Save original eWiseApply
        template<unsigned int descr, typename... Args>
        auto eWiseApply(Args&&... args)
            -> decltype(grb::eWiseApply<descr>(std::forward<Args>(args)...)) {
            return grb::eWiseApply<descr>(std::forward<Args>(args)...);
        }
        
        template<typename... Args>
        auto eWiseApply(Args&&... args)
            -> decltype(grb::eWiseApply(std::forward<Args>(args)...)) {
            return grb::eWiseApply(std::forward<Args>(args)...);
        }
        
        // Save original foldl
        template<unsigned int descr, typename... Args>
        auto foldl(Args&&... args)
            -> decltype(grb::foldl<descr>(std::forward<Args>(args)...)) {
            return grb::foldl<descr>(std::forward<Args>(args)...);
        }
        
        template<typename... Args>
        auto foldl(Args&&... args)
            -> decltype(grb::foldl(std::forward<Args>(args)...)) {
            return grb::foldl(std::forward<Args>(args)...);
        }
        
        // Save original dot
        template<unsigned int descr, typename... Args>
        auto dot(Args&&... args)
            -> decltype(grb::dot<descr>(std::forward<Args>(args)...)) {
            return grb::dot<descr>(std::forward<Args>(args)...);
        }
        
        template<typename... Args>
        auto dot(Args&&... args)
            -> decltype(grb::dot(std::forward<Args>(args)...)) {
            return grb::dot(std::forward<Args>(args)...);
        }
    }
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
    
    // GraphBLAS Matrix type detection - explicit common cases
    if (std::is_same<T, grb::Matrix<double>>::value) return "Matrix<double>";
    if (std::is_same<T, grb::Matrix<float>>::value) return "Matrix<float>";
    if (std::is_same<T, grb::Matrix<int>>::value) return "Matrix<int>";
    if (std::is_same<T, grb::Matrix<unsigned int>>::value) return "Matrix<unsigned int>";
    if (std::is_same<T, grb::Matrix<long>>::value) return "Matrix<long>";
    
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

// Helper for printing argument types
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
    
    // Print the type name and size if available
    std::cout << type_name << getSizeString<typename std::remove_reference<T>::type>(arg);
    
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

// Now redefine the functions in the grb namespace with tracing
namespace grb {
    // Override eWiseApply with descriptor
    template<unsigned int descr, typename... Args>
    auto eWiseApply(Args&&... args)
        -> decltype(original::eWiseApply<descr>(std::forward<Args>(args)...)) {
        
        std::string descriptor_name = std::to_string(descr);
        if (descr == descriptors::dense) descriptor_name = "dense";
        if (descr == descriptors::structural) descriptor_name = "structural";
        
        std::cout << "[TRACING] Entering function: eWiseApply<" << descriptor_name << "> with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::eWiseApply<descr>(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: eWiseApply<" << descriptor_name << "> (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Override eWiseApply without descriptor
    template<typename... Args>
    auto eWiseApply(Args&&... args)
        -> decltype(original::eWiseApply(std::forward<Args>(args)...)) {
        
        std::cout << "[TRACING] Entering function: eWiseApply with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::eWiseApply(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: eWiseApply (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Override foldl with descriptor
    template<unsigned int descr, typename... Args>
    auto foldl(Args&&... args)
        -> decltype(original::foldl<descr>(std::forward<Args>(args)...)) {
        
        std::string descriptor_name = std::to_string(descr);
        if (descr == descriptors::dense) descriptor_name = "dense";
        if (descr == descriptors::structural) descriptor_name = "structural";
        
        std::cout << "[TRACING] Entering function: foldl<" << descriptor_name << "> with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::foldl<descr>(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: foldl<" << descriptor_name << "> (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Override foldl without descriptor
    template<typename... Args>
    auto foldl(Args&&... args)
        -> decltype(original::foldl(std::forward<Args>(args)...)) {
        
        std::cout << "[TRACING] Entering function: foldl with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::foldl(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: foldl (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Override dot with descriptor
    template<unsigned int descr, typename... Args>
    auto dot(Args&&... args)
        -> decltype(original::dot<descr>(std::forward<Args>(args)...)) {
        
        std::string descriptor_name = std::to_string(descr);
        if (descr == descriptors::dense) descriptor_name = "dense";
        if (descr == descriptors::structural) descriptor_name = "structural";
        
        std::cout << "[TRACING] Entering function: dot<" << descriptor_name << "> with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::dot<descr>(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: dot<" << descriptor_name << "> (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
    
    // Override dot without descriptor
    template<typename... Args>
    auto dot(Args&&... args)
        -> decltype(original::dot(std::forward<Args>(args)...)) {
        
        std::cout << "[TRACING] Entering function: dot with " 
                  << sizeof...(args) << " arguments" << std::endl;
        
        printArgTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = original::dot(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: dot (took " 
                  << duration.count() << "μs)" << std::endl;
        
        return result;
    }
}

#endif // _GRB_ENABLE_TRACING

