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

// STEP 1: Define our decoration machinery before including original header
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
        return type_name;
    }
    
    // C++11 compatible argument type printing
    // Base case for recursion
    void printArgumentTypesHelper() const {
        // End of recursion - do nothing
    }
    
    // Recursive case - print one argument type and continue
    template<typename T, typename... Args>
    void printArgumentTypesHelper(T&& arg, Args&&... args) const {
        std::cout << getTypeName<typename std::decay<T>::type>() << " ";
        printArgumentTypesHelper(std::forward<Args>(args)...);
    }
    
    // Entry point for argument type printing
    template<typename... Args>
    void printArgumentTypes(Args&&... args) const {
        std::cout << "[DEBUG] Argument types: ";
        printArgumentTypesHelper(std::forward<Args>(args)...);
        std::cout << std::endl;
    }
    
public:
    DecoratedFunction(ImplFunc impl_func, std::string func_name) 
        : impl(std::move(impl_func)), name(std::move(func_name)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) const 
        -> decltype(impl(std::forward<Args>(args)...)) {
        
        std::cout << "[DEBUG] Entering function: " << name << " with " << sizeof...(args) << " arguments" << std::endl;
        printArgumentTypes(std::forward<Args>(args)...);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = impl(std::forward<Args>(args)...);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[DEBUG] Exiting function: " << name << " (took " << duration.count() << "μs)" << std::endl;
        
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

// Function pointer wrappers - these give us concrete types
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

