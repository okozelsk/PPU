using ILGPU;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;

namespace PPU4ILGPU
{

    /// <summary>
    /// Allocates available GPU devices among CPU threads using simple load balancing.
    /// Offers various modes of operation to suit different needs (debugging, testing of pure CPU versions of methods, etc.).
    /// </summary>
    /// <remarks>The <see cref="GPUAllocator"/> class is designed to manage GPU resources efficiently
    /// for computational tasks. It supports multiple modes of operation, allowing users to select the most appropriate
    /// resource allocation strategy for their needs.  The allocator ensures thread-safe access to resources and
    /// tracks the number of active jobs to optimize resource utilization. Users must call <see cref="Acquire"/> to
    /// obtain an accelerator and <see cref="Release"/> to release it after use. Failure to call <see cref="Release"/>
    /// may result in resource leaks or inconsistent state.  This class implements <see cref="IDisposable"/> to release
    /// resources when the allocator is no longer needed. Ensure that <see cref="Dispose"/> is called to clean up
    /// resources properly.</remarks>
    public sealed class GPUAllocator : IDisposable
    {
        //Enums
        /// <summary>
        /// Specifies the operational mode for workload distribution and hardware utilization.
        /// </summary>
        /// <remarks>The <see cref="Mode"/> enumeration defines various modes for utilizing hardware
        /// accelerators, such as GPUs or CPUs, to execute workloads. Each mode determines how resources are allocated
        /// and whether GPU acceleration is used. This can be useful for optimizing performance, testing, or
        /// debugging.</remarks>
        public enum Mode
        {
            /// <summary>
            /// Standard mode with GPU acceleration (if available). The workload is
            /// balanced among all available GPU devices.
            /// </summary>
            Standard,
            /// <summary>
            /// Exclusive utilization of the most powerful GPU device (if available).
            /// </summary>
            MostPowerfulGPU,
            /// <summary>
            /// Exclusive utilization of the least powerful GPU device (if available).
            /// </summary>
            LeastPowerfulGPU,
            /// <summary>
            /// Simulates situation, when no GPU device is available.
            /// Useful for testing pure CPU versions of methods.
            /// </summary>
            NoAccelerator,
            /// <summary>
            /// CPU accelerator simulating GPU is used instead of real GPU accelerator.
            /// Useful for debugging of GPU code.
            /// </summary>
            CPUAccelerator
        }

        /// <summary>
        /// Represents a pairing of a GPU accelerator with the number of jobs currently booked on it.
        /// </summary>
        /// <remarks>This class is used to track the association between a GPU accelerator and the number
        /// of jobs assigned to it. It provides properties to access the accelerator and manage the job count.</remarks>
        private class AccelJobsPair
        {
            internal GPUWrappedAccelerator WrappedAccel { get; }
            internal int NumOfBookedJobs { get; set; } = 0;

            internal AccelJobsPair(GPUWrappedAccelerator wrappedAccel, int numOfBookedJobs)
            {
                WrappedAccel = wrappedAccel;
                NumOfBookedJobs = numOfBookedJobs;
            }

        }

        /// <summary>
        /// Gets the total number of jobs that have been booked.
        /// </summary>
        public int NumOfBookedJobs { get; private set; }

        /// <summary>
        /// Gets the peak number of booked jobs recorded during the application's runtime.
        /// </summary>
        public int BookedJobsPeak { get; private set; }


        private readonly object _syncRoot;
        private Mode _mode;
        private readonly Context _context;
        private readonly List<AccelJobsPair> _GPUs;
        private readonly AccelJobsPair? _CPU;
        private int _isDisposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPUAllocator"/> class, which manages GPU and CPU resources for
        /// computational tasks, prioritizing GPU accelerators based on their performance capabilities.
        /// </summary>
        /// <remarks>This constructor initializes GPU and CPU accelerators by detecting available devices
        /// in the system. GPU accelerators are sorted from the most powerful to the least powerful based on their
        /// performance score. The allocator supports CUDA and OpenCL GPU devices, as well as CPU devices for debugging.</remarks>
        /// <param name="mode">The allocation mode to use. Defaults to <see cref="Mode.Standard"/>.</param>
        public GPUAllocator(Mode mode = Mode.Standard)
        {
            _syncRoot = new();
            _mode = mode;
            // Initialize GPU resources
            //Context
            _context = Context.Create(builder =>
            {
                builder.Default()
                       .Optimize(OptimizationLevel.O2)
                       .PTXBackend(PTXBackendMode.Enhanced)
                       .EnableAlgorithms();
            });
            // List of GPU accelerators
            _GPUs = new(_context.Devices.Length);
            _CPU = null;
            foreach (var device in _context.Devices)
            {
                if (device.AcceleratorType == AcceleratorType.Cuda || device.AcceleratorType == AcceleratorType.OpenCL)
                {
                    // Create an accelerator for GPU device
                    GPUWrappedAccelerator wAcc = new(_context, device);
                    _GPUs.Add(new(wAcc, 0));
                }
                else if (device.AcceleratorType == AcceleratorType.CPU)
                {
                    // Create an accelerator for CPU device
                    _CPU = new(new(_context, device), 0);
                }
            }
            //Sort GPU accelerators from the most powerful to the least powerful
            _GPUs.Sort((a, b) => b.WrappedAccel.PowerScore.CompareTo(a.WrappedAccel.PowerScore));
            _isDisposed = 0;
            return;
        }

        /// <summary>
        /// Sets the operational mode of the allocator.
        /// </summary>
        /// <param name="mode">The new mode to set. This value determines the behavior of the allocator.</param>
        /// <exception cref="ObjectDisposedException"></exception>
        public void SetMode(Mode mode)
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            lock (_syncRoot)
            {
                _mode = mode;
                return;
            }
        }

        private void IncNumOfBookedJobs()
        {
            ++NumOfBookedJobs;
            BookedJobsPeak = Math.Max(BookedJobsPeak, NumOfBookedJobs);
            return;
        }

        /// <summary>
        /// Acquires an available accelerator based on the current mode and increments its job count.
        /// </summary>
        /// <remarks>This method selects and returns an accelerator according to the configured mode:
        /// <list type="bullet"> <item> <description><see cref="Mode.NoAccelerator"/>: Returns <see
        /// langword="null"/>.</description> </item> <item> <description><see cref="Mode.CPUAccelerator"/>: Returns the
        /// CPU accelerator, if available.</description> </item> <item> <description><see
        /// cref="Mode.MostPowerfulGPU"/>: Returns the most powerful GPU accelerator, if available.</description>
        /// </item> <item> <description><see cref="Mode.LeastPowerfulGPU"/>: Returns the least powerful GPU accelerator,
        /// if available.</description> </item> <item> <description>Standard mode: Returns the GPU accelerator with the
        /// fewest booked jobs, if available.</description> </item> </list> If no accelerators are available, the method
        /// returns <see langword="null"/>.</remarks>
        /// <returns>A <see cref="GPUWrappedAccelerator"/> representing the selected accelerator, or <see langword="null"/> if no
        /// suitable accelerator is available.</returns>
        public GPUWrappedAccelerator? Acquire()
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            lock (_syncRoot)
            {
                // Special cases
                if (_mode == Mode.NoAccelerator)
                {
                    return null;
                }
                else if (_mode == Mode.CPUAccelerator)
                {
                    if (_CPU != null)
                    {
                        ++_CPU.NumOfBookedJobs;
                        IncNumOfBookedJobs();
                    }
                    return _CPU?.WrappedAccel;
                }
                else if (_GPUs.Count == 0)
                {
                    // No GPU accelerators available
                    return null;
                }
                //GPU accelerators available, get the one according to mode
                int idx = 0;
                if (_mode == Mode.MostPowerfulGPU)
                {
                    idx = 0;
                }
                else if (_mode == Mode.LeastPowerfulGPU)
                {
                    idx = _GPUs.Count - 1;
                }
                else
                {
                    // Standard mode
                    int minNumOfBookedJobs = int.MaxValue;
                    // Find the GPU with the minimum number of booked jobs
                    // and if there are multiple ones, select the one with the highest power score
                    for (int i = 0; i < _GPUs.Count; i++)
                    {
                        int numOfBookedJobs = _GPUs[i].NumOfBookedJobs;
                        if (numOfBookedJobs < minNumOfBookedJobs)
                        {
                            minNumOfBookedJobs = numOfBookedJobs;
                            idx = i;
                        }
                    }
                }
                // Return the selected accelerator
                GPUWrappedAccelerator accel = _GPUs[idx].WrappedAccel;
                ++_GPUs[idx].NumOfBookedJobs;
                IncNumOfBookedJobs();
                return accel;
            }
        }

        /// <summary>
        /// Releases a previously acquired GPU accelerator, decrementing its booked job count.
        /// </summary>
        /// <remarks>This method ensures that the job count for the specified accelerator is decremented
        /// correctly. If the accelerator is not found or the job count is already zero, an exception is thrown to
        /// indicate an invalid state. Thread safety is maintained using a synchronization lock.</remarks>
        /// <param name="accel">The GPU accelerator to release. Must have been previously acquired.</param>
        /// <exception cref="ApplicationException">Thrown if the release operation detects an inconsistent number of acquire and release calls.</exception>
        /// <exception cref="ArgumentException">Thrown if the specified <paramref name="accel"/> is not found in the list of accelerators.</exception>
        public void Release(GPUWrappedAccelerator accel)
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            lock (_syncRoot)
            {
                if (_CPU != null && accel == _CPU.WrappedAccel)
                {
                    if (_CPU.NumOfBookedJobs <= 0)
                    {
                        throw new ApplicationException("Detected inconsistent Acquire/Release calls.");
                    }
                    --_CPU.NumOfBookedJobs;
                }
                else
                {
                    int idx = _GPUs.FindIndex(x => x.WrappedAccel == accel);
                    if (idx < 0)
                    {
                        throw new ArgumentException("GPU accelerator not found in the list of GPU accelerators.", nameof(accel));
                    }
                    if (_GPUs[idx].NumOfBookedJobs <= 0)
                    {
                        throw new ApplicationException("Detected inconsistent Acquire/Release calls.");
                    }
                    --_GPUs[idx].NumOfBookedJobs;
                }
                NumOfBookedJobs = Math.Max(0, NumOfBookedJobs - 1);
            }
            return;
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
                    lock (_syncRoot)
                    {
                        foreach (var item in _GPUs)
                        {
                            item.WrappedAccel.Dispose();
                        }
                        _GPUs.Clear();
                        _CPU?.WrappedAccel.Dispose();
                        _context.Dispose();
                    }
                }
            }
            return;
        }

    }
}
