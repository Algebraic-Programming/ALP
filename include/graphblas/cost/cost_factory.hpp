#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <vector>
#include <type_traits>

// Define a compile-time toggle
#ifndef _GRB_ENABLE_TRACING
#define _GRB_ENABLE_TRACING 1  // Set to 1 for this example
#endif

// STEP 1: Define the decoration machinery before including original header
#if _GRB_ENABLE_TRACING
// Decorator class definition
template<typename ImplFunc>
class DecoratedFunction {
private:
    ImplFunc impl;
    std::string name;
    
    // Helper to get type names
    template<typename T>
    std::string getTypeName() const {
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
        
        // GraphBLAS Operator Instance detection (with parentheses)
        // Add operator
        if (std::is_same<T, grb::operators::add<double>>::value) return "operators::add<double>";
        if (std::is_same<T, grb::operators::add<float>>::value) return "operators::add<float>";
        if (std::is_same<T, grb::operators::add<int>>::value) return "operators::add<int>";
        
        // Mul operator
        if (std::is_same<T, grb::operators::mul<double>>::value) return "operators::mul<double>";
        if (std::is_same<T, grb::operators::mul<float>>::value) return "operators::mul<float>";
        if (std::is_same<T, grb::operators::mul<int>>::value) return "operators::mul<int>";
        
        // Min operator
        if (std::is_same<T, grb::operators::min<double>>::value) return "operators::min<double>";
        if (std::is_same<T, grb::operators::min<float>>::value) return "operators::min<float>";
        if (std::is_same<T, grb::operators::min<int>>::value) return "operators::min<int>";
        
        // Max operator
        if (std::is_same<T, grb::operators::max<double>>::value) return "operators::max<double>";
        if (std::is_same<T, grb::operators::max<float>>::value) return "operators::max<float>";
        if (std::is_same<T, grb::operators::max<int>>::value) return "operators::max<int>";
        
        // Fallback string matching for operators if the exact type isn't matched above
        if (type_name.find("operators::add") != std::string::npos) {
            if (type_name.find("double") != std::string::npos) return "operators::add<double>";
            if (type_name.find("float") != std::string::npos) return "operators::add<float>";
            if (type_name.find("int") != std::string::npos) return "operators::add<int>";
            return "operators::add<...>";
        }
        
        if (type_name.find("operators::mul") != std::string::npos) {
            if (type_name.find("double") != std::string::npos) return "operators::mul<double>";
            if (type_name.find("float") != std::string::npos) return "operators::mul<float>";
            if (type_name.find("int") != std::string::npos) return "operators::mul<int>";
            return "operators::mul<...>";
        }
        
        if (type_name.find("operators::min") != std::string::npos) {
            if (type_name.find("double") != std::string::npos) return "operators::min<double>";
            if (type_name.find("float") != std::string::npos) return "operators::min<float>";
            if (type_name.find("int") != std::string::npos) return "operators::min<int>";
            return "operators::min<...>";
        }
        
        if (type_name.find("operators::max") != std::string::npos) {
            if (type_name.find("double") != std::string::npos) return "operators::max<double>";
            if (type_name.find("float") != std::string::npos) return "operators::max<float>";
            if (type_name.find("int") != std::string::npos) return "operators::max<int>";
            return "operators::max<...>";
        }
        
        // Semiring detection
        if (type_name.find("Semiring<") != std::string::npos) {
            if (type_name.find("add") != std::string::npos && type_name.find("mul") != std::string::npos) {
                if (type_name.find("double") != std::string::npos) return "Semiring<add,mul,double>";
                if (type_name.find("float") != std::string::npos) return "Semiring<add,mul,float>";
                if (type_name.find("int") != std::string::npos) return "Semiring<add,mul,int>";
                return "Semiring<add,mul,...>";
            }
            
            if (type_name.find("min") != std::string::npos && type_name.find("plus") != std::string::npos) {
                return "Semiring<min,plus,...>";
            }
            
            if (type_name.find("max") != std::string::npos && type_name.find("mul") != std::string::npos) {
                return "Semiring<max,mul,...>";
            }
            
            return "Semiring<...>";
        }
        
        // Special named semirings
        if (type_name.find("mul_max") != std::string::npos) {
            if (type_name.find("double") != std::string::npos) return "mul_max_double";
            if (type_name.find("float") != std::string::npos) return "mul_max_float";
            if (type_name.find("int") != std::string::npos) return "mul_max_int";
            return "mul_max_semiring";
        }
        
        // Generic fallbacks for other GraphBLAS types
        if (type_name.find("Vector") != std::string::npos) {
            return "Vector<...>";
        }
        
        if (type_name.find("Matrix") != std::string::npos) {
            return "Matrix<...>";
        }
        
        if (type_name.find("operators::") != std::string::npos) {
            return "operators::...";
        }
        
        // Return the raw type name if nothing matched
        return type_name;
    }
    
    // C++11 compatible argument type printing
    // Base case for recursion
    void printArgumentTypesHelper() const {
        // End of recursion - do nothing
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

    // Helper to safely get size if available (vector types)
    template<typename T>
    typename std::enable_if<has_grb_size<T>::value, std::string>::type
    getSizeString(const T& arg) const {
        try {
            return "[size=" + std::to_string(grb::size(arg)) + "] ";
        } catch(...) {
            return " ";
        }
    }

    // Helper for types that don't support size (non-vector types)
    template<typename T>
    typename std::enable_if<!has_grb_size<T>::value, std::string>::type
    getSizeString(const T&) const {
        return " ";
    }

    // Single implementation that handles all argument types
    template<typename T, typename... Args>
    void printArgumentTypesHelper(T&& arg, Args&&... args) const {
        // Get the type name
        std::string type_name = getTypeName<typename std::decay<T>::type>();
        
        // Print the type name and size if available
        std::cout << type_name << getSizeString<typename std::remove_reference<T>::type>(arg);
        
        // Continue with remaining arguments
        printArgumentTypesHelper(std::forward<Args>(args)...);
    }
    
    // Entry point for argument type printing
    template<typename... Args>
    void printArgumentTypes(Args&&... args) const {
        std::cout << "[TRACING] Argument types: ";
        printArgumentTypesHelper(std::forward<Args>(args)...);
        std::cout << std::endl;
    }
    
public:
    DecoratedFunction(ImplFunc impl_func, std::string func_name) 
        : impl(std::move(impl_func)), name(std::move(func_name)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) const 
        -> decltype(impl(std::forward<Args>(args)...)) {
        
        std::cout << "[TRACING] Entering function: " << name << " with " << sizeof...(args) << " arguments" << std::endl;
        printArgumentTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = impl(std::forward<Args>(args)...);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[TRACING] Exiting function: " << name << " (took " << duration.count() << "μs)" << std::endl;

        return result;
    }
};

// Factory function
template<typename Func>
DecoratedFunction<Func> make_decorated_function(Func func, const std::string& name) {
    return DecoratedFunction<Func>(func, name);
}

// Create a forwarding functor that explicitly wraps the function
template<typename FuncPtr>
struct ForwardingFunctor {
    FuncPtr func_ptr;
    
    ForwardingFunctor(FuncPtr ptr) : func_ptr(ptr) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(func_ptr(std::forward<Args>(args)...)) {
        return func_ptr(std::forward<Args>(args)...);
    }
};

// Helper to create a forwarding functor
template<typename FuncPtr>
ForwardingFunctor<FuncPtr> make_forwarder(FuncPtr func_ptr) {
    return ForwardingFunctor<FuncPtr>(func_ptr);
}

// Function pointer wrappers - these give concrete types
template<typename... Args>
decltype(grb::eWiseApply(std::declval<Args>()...))
eWiseApply_wrapper(Args&&... args) {
    return grb::eWiseApply(std::forward<Args>(args)...);
}

template<typename... Args>
decltype(grb::foldl(std::declval<Args>()...))
foldl_wrapper(Args&&... args) {
    return grb::foldl(std::forward<Args>(args)...);
}

template<typename... Args>
decltype(grb::dot(std::declval<Args>()...))
dot_wrapper(Args&&... args) {
    return grb::dot(std::forward<Args>(args)...);
}

// Step 2: Create decorated replacements for specific functions
namespace grb {
    // Decorated eWiseApply
    template<typename... Args>
    auto decorated_eWiseApply(Args&&... args) 
        -> decltype(grb::eWiseApply(std::forward<Args>(args)...)) {
        // This works with a single variadic template
        // We create a forwarder around the wrapper function
        static auto decorated = make_decorated_function(
            make_forwarder(&eWiseApply_wrapper<Args...>),
            "eWiseApply"
        );
        return decorated(std::forward<Args>(args)...);
    }

    // Decorated foldl
    template<typename... Args>
    auto decorated_foldl(Args&&... args) 
        -> decltype(grb::foldl(std::forward<Args>(args)...)) {
        static auto decorated = make_decorated_function(
            make_forwarder(&foldl_wrapper<Args...>),
            "foldl"
        );
        return decorated(std::forward<Args>(args)...);
    }

    // Decorated dot
    template<typename... Args>
    auto decorated_dot(Args&&... args) 
        -> decltype(grb::dot(std::forward<Args>(args)...)) {
        static auto decorated = make_decorated_function(
            make_forwarder(&dot_wrapper<Args...>),
            "dot"
        );
        return decorated(std::forward<Args>(args)...);
    }
}

// Step 3: Replace the original functions with preprocessor macros
// This avoids circular dependencies while keeping the same function names
#define eWiseApply decorated_eWiseApply
#define foldl decorated_foldl
#define dot decorated_dot

#endif // _GRB_ENABLE_TRACING

