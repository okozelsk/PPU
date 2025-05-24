using ILGPU;
using ILGPU.Runtime;

namespace PPU4ILGPU
{
    /// <summary>
    /// Represents a GPU accelerator wrapper that provides functionality for managing GPU resources, including kernel
    /// caching, stream pooling, and thread synchronization for GPU operations.
    /// </summary>
    /// <remarks>This class is designed to facilitate GPU operations in a multithreaded environment. It
    /// ensures thread safety for GPU operations, particularly for OpenCL accelerators, by requiring the use of the <see
    /// cref="ThreadLock"/> object. CUDA accelerators are generally thread-safe and may not require this locking
    /// mechanism.</remarks>
    public class GPUWrappedAccelerator : IDisposable
    {
        /// <summary>
        /// A thread synchronization object used to lock critical sections of code.
        /// </summary>
        /// <remarks>Use this object to ensure thread safety when accessing shared resources. It is
        /// recommended to use a <c>lock</c> statement with this object to prevent race conditions in multithreaded
        /// environments.</remarks>
        public readonly object ThreadLock;

        /// <summary>
        /// Gets the <see cref="Accelerator"/> object associated with this instance.
        /// </summary>
        public Accelerator AccelObj { get; }

        /// <summary>
        /// Gets the cache of GPU kernels available for use.
        /// </summary>
        public GPUKernelCache Kernels { get; }

        /// <summary>
        /// Gets the pool of GPU streams used for managing and reusing GPU resources efficiently.
        /// </summary>
        public GPUStreamPool StreamPool { get; }

        /// <summary>
        /// Power score of the accelerator.
        /// Currently it is calculated as number of multiprocessors times maximum number of threads per group.
        /// Times ClockRate is not used because it is not publicly and consistently provided ILGPU Device property.
        /// </summary>
        public double PowerScore { get; }

        private int _isDisposed;


        /// <summary>
        /// Initializes a new instance of the <see cref="GPUWrappedAccelerator"/> class, wrapping a GPU accelerator for
        /// use with the specified context and device.
        /// </summary>
        /// <param name="context">The context in which the accelerator will operate. Cannot be <see langword="null"/>.</param>
        /// <param name="device">The device used to create the accelerator. Cannot be <see langword="null"/>.</param>
        /// <exception cref="ArgumentNullException"></exception>
        public GPUWrappedAccelerator(Context context, Device device)
        {
            ArgumentNullException.ThrowIfNull(context);
            ArgumentNullException.ThrowIfNull(device);
            AccelObj = device.CreateAccelerator(context);
            ThreadLock = new();
            Kernels = new();
            StreamPool = new(AccelObj);
            PowerScore = (AccelObj.NumMultiprocessors * AccelObj.MaxNumThreadsPerGroup);
            _isDisposed = 0;
            return;
        }

        /// <summary>
        /// Calculates and returns the kernel configuration required to process a specified number of data elements.
        /// </summary>
        /// <remarks>The method determines the optimal kernel configuration based on the number of data
        /// elements and the maximum allowable groups. If the number of data elements exceeds the maximum number of
        /// groups and <paramref name="gridStrideLoop"/> is <see langword="true"/>,  the method will limit the number of
        /// groups to the maximum and rely on a grid-stride loop for processing.</remarks>
        /// <param name="numOfDataElements">The total number of data elements to be processed. Must be a non-negative value.</param>
        /// <param name="gridStrideLoop">A boolean value indicating whether to use a grid-stride loop if the number of data elements exceeds the
        /// maximum number of groups. If <see langword="true"/>, the kernel will use the maximum number of groups and
        /// rely on a grid-stride loop to process all elements. If <see langword="false"/>, the kernel will use the
        /// number of groups required to cover all data elements, even if it exceeds the maximum number of groups.</param>
        /// <returns>A <see cref="KernelConfig"/> structure containing the number of kernel groups and the group size required
        /// for processing the data.</returns>
        /// <exception cref="ObjectDisposedException"></exception>
        public KernelConfig GetKernelConfig(long numOfDataElements, bool gridStrideLoop = false)
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            (Index1D maxGroups, Index1D groupSize) = AccelObj.MaxNumGroupsExtent;
            // Compute number of necessary groups to cover all the data (round up)
            Index1D dataGroups = (Index1D)((numOfDataElements + groupSize.LongSize - 1) / groupSize.LongSize);
            //Determine what number of groups to use
            Index1D kernelGroups = dataGroups <= maxGroups ? dataGroups : gridStrideLoop ? maxGroups : dataGroups;
            return (kernelGroups, groupSize);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
            return;
        }

        private void Dispose(bool disposing)
        {
            if (Interlocked.CompareExchange(ref _isDisposed, 1, 0) == 0)
            {
                if (disposing)
                {
                    lock (ThreadLock)
                    {
                        StreamPool.Dispose();
                        AccelObj.Dispose();
                    }
                }
            }
            return;
        }
    }
}
