using System.Collections.Concurrent;

namespace PPU4ILGPU
{
    /// <summary>
    /// Provides a thread-safe cache for storing and retrieving GPU kernel delegates by name.
    /// </summary>
    /// <remarks>This class is designed to manage GPU kernel delegates, allowing efficient reuse of compiled
    /// kernels. Kernels are identified by unique string keys, and if a requested kernel is not found in the cache, it
    /// is created using a provided factory function, stored in the cache, and then returned.</remarks>
    public sealed class GPUKernelCache
    {
        private readonly ConcurrentDictionary<string, Delegate> _cache;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPUKernelCache"/> class.
        /// </summary>
        /// <remarks>This constructor creates an empty cache for storing GPU kernel objects. Use this
        /// class to manage and reuse GPU kernels efficiently.</remarks>
        public GPUKernelCache()
        {
            _cache = new();
            return;
        }

        /// <summary>
        /// Retrieves a cached kernel delegate associated with the specified key, or adds a new one if it does not
        /// exist.
        /// </summary>
        /// <remarks>This method ensures that a kernel delegate is either retrieved from the cache or
        /// compiled and added to the cache in a thread-safe manner.</remarks>
        /// <typeparam name="T">The type of the delegate representing the kernel.</typeparam>
        /// <param name="key">The unique key used to identify the kernel in the cache. Cannot be <see langword="null"/> or empty.</param>
        /// <param name="compileKernelFn">A function that compiles and returns the kernel delegate if it is not already cached.  This function is
        /// invoked only when the kernel is not found in the cache.</param>
        /// <returns>The kernel delegate of type <typeparamref name="T"/> associated with the specified key.  If the kernel was
        /// not already cached, the result of <paramref name="compileKernelFn"/> is cached and returned.</returns>
        public T GetOrAddKernel<T>(string key, Func<T> compileKernelFn) where T : Delegate
        {
            if (_cache.TryGetValue(key, out Delegate? cachedKernel))
            {
                return (T)cachedKernel;
            }
            else
            {
                T kernel = compileKernelFn();
                _cache[key] = kernel;
                return kernel;
            }
        }

    }

}
