using ILGPU;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using PPU4ILGPU;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TutorialApp
{
    /// <summary>
    /// Demonstrates the usage of GPU allocation and reservation mechanisms for parallel processing tasks.
    /// </summary>
    /// <remarks>This class provides an example of how to use the <see cref="GPUAllocator"/> to manage GPU
    /// resources for parallel operations. It shows reserving a specific GPU for exclusive
    /// use and releasing it back among available GPUs.</remarks>
    public class AllocatorGPUReservationExample
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="AllocatorGPUReservationExample"/> class.
        /// </summary>
        /// <remarks>This constructor sets up the example by configuring the GPU allocation mode to  <see
        /// cref="GPUAllocator.Mode.Standard"/>, which uses all available GPU devices. It also outputs initialization
        /// messages to the console.</remarks>
        public AllocatorGPUReservationExample()
        {
            Console.WriteLine("*******************");
            Console.WriteLine("* Example started *");
            Console.WriteLine("*******************");
            //Set the GPU allocation mode to Standard to use the default GPU allocation mode where all GPU devices are used
            GPU.Allocator.SetMode(GPUAllocator.Mode.Standard);
            Console.WriteLine();
        }

        /// <summary>
        /// Simulates a job execution, utilizing a GPU accelerator if available, or falling back to CPU processing.
        /// </summary>
        /// <remarks>If a GPU accelerator is available, it is acquired and used for processing. The method
        /// ensures thread safety by locking the GPU accelerator during its usage. If no GPU is available, the job is
        /// processed on the CPU. The method logs the processing details, including whether the GPU or CPU was used, and
        /// the total execution time.</remarks>
        /// <param name="threadName">The name of the thread executing the job. This is used for logging purposes to identify the source of the
        /// output.</param>
        private static void JobSimulation(string threadName)
        {
            Stopwatch sw = new();
            //Start measuring time
            sw.Start();

            //Try to acquire a GPU accelerator for processing
            GPUWrappedAccelerator? a = GPU.Allocator.Acquire();
            if (a == null)
            {
                // No GPU available, fallback to CPU processing
                Console.WriteLine($"[{threadName}] started processing on CPU...");
                Thread.Sleep(100); // Simulate some CPU work
            }
            else
            {
                // GPU is available, use it for processing
                Console.WriteLine($"[{threadName}] started processing on device {a.AccelObj.Device.Name}...");
                //Lock the accelerator to ensure thread safety
                lock (a.ThreadLock)
                {
                    Thread.Sleep(100); // Simulate some CPU work
                    GPU.Allocator.Release(a);
                }
            }
            //Stop measuring time
            sw.Stop();
            Console.WriteLine($"[{threadName}] finished in {sw.ElapsedMilliseconds} ms.");
            return;
        }

        /// <summary>
        /// Executes a sequence of six parallel operations, each performing fictive job.
        /// </summary>
        private void ExecuteParallelSequence()
        {
            Stopwatch sw = new();
            Console.WriteLine();
            Console.WriteLine($"Parallel sequence started with {nameof(GPUAllocator)} operation mode {Enum.GetName(typeof(GPUAllocator.Mode), GPU.Allocator.CurrentMode())}...");
            // Start measuring time
            sw.Start();
            // Execute the neighbor sum calculations in parallel
            Parallel.Invoke(
                () => { JobSimulation($"T1"); },
                () => { JobSimulation($"T2"); },
                () => { JobSimulation($"T3"); },
                () => { JobSimulation($"T4"); },
                () => { JobSimulation($"T5"); },
                () => { JobSimulation($"T6"); }
                );
            // Stop measuring time
            sw.Stop();
            Console.WriteLine($"Parallel sequence completed in {sw.ElapsedMilliseconds} ms.");
            return;
        }

        /// <summary>
        /// Executes a sequence of operations utilizing available GPUs, demonstrating GPU allocation, reservation, and
        /// release.
        /// </summary>
        /// <remarks>This method performs the following steps: <list type="number"> <item> Lists all
        /// available GPUs, sorted from the most powerful to the least powerful. </item> <item> Executes a parallel
        /// sequence using all available GPUs. </item> <item> Attempts to reserve the most powerful GPU for exclusive
        /// use and executes another parallel sequence with the remaining GPUs. </item> <item> Releases the reserved GPU
        /// and executes a final parallel sequence with all GPUs available again. </item> If no GPUs are available at
        /// the start, the method exits early. The method also handles scenarios where GPU reservation or release
        /// fails.</remarks>
        public void Run()
        {
            Console.WriteLine("All available GPUs sorted from the most powerfull to the least powerfull");
            List<GPUWrappedAccelerator> availableGPUs = GPU.Allocator.GetAvailableGPUs();
            foreach (var wa in availableGPUs)
            {
                Console.WriteLine($"- {wa.AccelObj.Device.Name} ({wa.AccelObj.Device.AcceleratorType})");
            }

            if(availableGPUs.Count == 0)
            {
                Console.WriteLine("No GPUs available. Exiting...");
                return;
            }

            Console.WriteLine();
            Console.WriteLine("Execution of parallel sequence with all GPUs available");
            ExecuteParallelSequence();

            Console.WriteLine();
            Console.WriteLine("Try to reserve the most powerful GPU for long term...");
            GPUWrappedAccelerator mpAccel = availableGPUs[0];
            if (!GPU.Allocator.ReserveGPU(mpAccel))
            {
                Console.WriteLine("Failed to reserve the most powerful GPU. Exiting...");
                return;
            }
            else
            {
                Console.WriteLine($"{mpAccel.AccelObj.Device.Name} is now reserved for exclusive use, which is preventing {nameof(GPUAllocator)} to allocate it.");
            }

            Console.WriteLine("Remaining GPUs available");
            foreach (var wa in GPU.Allocator.GetAvailableGPUs())
            {
                Console.WriteLine($"- {wa.AccelObj.Device.Name}");
            }
            Console.WriteLine("Execution of parallel sequence with remaining GPUs available");
            ExecuteParallelSequence();

            Console.WriteLine();
            Console.WriteLine($"Release reserved {mpAccel.AccelObj.Device.Name}...");
            if (!GPU.Allocator.ReleaseReservedGPU(mpAccel))
            {
                Console.WriteLine("Failed to reserve the most powerful GPU. Exiting...");
                return;
            }
            Console.WriteLine("Execution of parallel sequence with all GPUs available again");
            ExecuteParallelSequence();

            return;
        }

    }
}
