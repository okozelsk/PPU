namespace PPU4ILGPU
{
    /// <summary>
    /// Provides access to a globally shared instance of a GPU allocator, ensuring efficient management of GPU
    /// resources.
    /// </summary>
    /// <remarks>The <see cref="GPU"/> class is a singleton that provides a thread-safe, lazily initialized
    /// instance of  <see cref="GPUAllocator"/>. The allocator is automatically disposed when the application exits,
    /// ensuring  proper cleanup of GPU resources. <see cref="GPU"/> class cannot be instantiated directly.</remarks>
    public sealed class GPU
    {
        private static readonly Lazy<GPUAllocator> lazy = new(() => new GPUAllocator(GPUAllocator.Mode.Standard));
        private static readonly object _syncRoot = new();
        private static bool _onProcessExitRegistered = false;

        /// <summary>
        /// Gets the global GPU memory allocator instance.
        /// </summary>
        /// <remarks>The allocator is lazily initialized and ensures thread-safe access. It automatically
        /// cleans up resources when the application exits by handling the <see cref="AppDomain.ProcessExit"/>
        /// event.</remarks>
        public static GPUAllocator Allocator
        {
            get
            {
                if (!lazy.IsValueCreated)
                {
                    lock (_syncRoot)
                    {
                        if (!_onProcessExitRegistered)
                        {
                            // Register for process exit event to clean up resources
                            AppDomain.CurrentDomain.ProcessExit += OnProcessExit;
                            _onProcessExitRegistered = true;
                        }
                    }
                }
                return lazy.Value;
            }
        }

        /// <summary>
        /// Private constructor to prevent instantiation of this class.
        /// </summary>
        private GPU()
        {
        }

        /// <summary>
        /// Cleans up GPU resources when the process exits.
        /// </summary>
        private static void OnProcessExit(object? sender, EventArgs e)
        {
            if (lazy.IsValueCreated)
            {
                lazy.Value.Dispose();
            }
            return;
        }


    }
}
