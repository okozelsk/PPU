﻿using ILGPU;
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
    /// Demonstrates the use of GPU and CPU parallel processing for neighbor summation in six 2D arrays
    /// using different GPU allocation modes and measuring performance.
    /// This example does exactly the same as <see cref="AllocatorExample"/>, but for GPU operations
    /// uses thread-dedicated stream acquired from <see cref="GPUStreamPool"/>, instead of default stream.
    /// </summary>
    public class AllocatorNDStreamExample
    {
        private readonly float[,] _2DArrayOfFloats1;
        private readonly float[,] _2DArrayOfFloats2;
        private readonly float[,] _2DArrayOfFloats3;
        private readonly float[,] _2DArrayOfFloats4;
        private readonly float[,] _2DArrayOfFloats5;
        private readonly float[,] _2DArrayOfFloats6;

        /// <summary>
        /// Initializes a new instance of the <see cref="AllocatorNDStreamExample"/> class.
        /// </summary>
        /// <remarks>This constructor initializes six 2D arrays
        /// populated with random float values between 0.0 and 1.0.
        /// Constructor also reports available
        /// GPUs on host machine (for information only).</remarks>
        public AllocatorNDStreamExample()
        {
            Console.WriteLine("*******************");
            Console.WriteLine("* Example started *");
            Console.WriteLine("*******************");
            Random rand = new();
            int height = 600;
            int width = 800;
            _2DArrayOfFloats1 = new float[height, width];
            _2DArrayOfFloats2 = new float[height, width];
            _2DArrayOfFloats3 = new float[height, width];
            _2DArrayOfFloats4 = new float[height, width];
            _2DArrayOfFloats5 = new float[height, width];
            _2DArrayOfFloats6 = new float[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    _2DArrayOfFloats1[i, j] = (float)rand.NextDouble();
                    _2DArrayOfFloats2[i, j] = (float)rand.NextDouble();
                    _2DArrayOfFloats3[i, j] = (float)rand.NextDouble();
                    _2DArrayOfFloats4[i, j] = (float)rand.NextDouble();
                    _2DArrayOfFloats5[i, j] = (float)rand.NextDouble();
                    _2DArrayOfFloats6[i, j] = (float)rand.NextDouble();
                }
            }
            Console.WriteLine("Available GPUs");
            foreach (var wa in GPU.Allocator.GetAvailableGPUs())
            {
                Console.WriteLine($"- {wa.AccelObj.Device.Name} ({wa.AccelObj.Device.AcceleratorType})");
            }
            Console.WriteLine();
        }


        /// <summary>
        /// Computes the sum of neighboring elements for each element in a 2D array.
        /// GPU processing uses dedicated stream from <see cref="GPUStreamPool"/> on <see cref="GPUWrappedAccelerator"/>.
        /// </summary>
        /// <remarks>This method processes the input array using either CPU or GPU resources, depending on
        /// availability.  If a GPU is available, the computation is offloaded to the GPU for improved performance.
        /// Otherwise,  the computation is performed on the CPU using parallel processing.  The radius for neighbor
        /// summation is fixed at 5. Neighboring elements are considered only if they fall within the bounds of the
        /// input array.  The method is thread-safe and can be used in multi-threaded environments.</remarks>
        /// <param name="threadName">The name of the thread or task performing the operation, used for logging purposes.</param>
        /// <param name="input">A 2D array of floating-point numbers representing the input data. Must not be null.</param>
        /// <returns>A 2D array of the same dimensions as <paramref name="input"/>, where each element contains the sum of its
        /// neighbors within a predefined radius, excluding the element itself.</returns>
        private float[,] NeighborSum(string threadName, float[,] input)
        {
            const int radius = 5; // Radius for neighbor summation
            float[,] result = new float[input.GetLength(0), input.GetLength(1)];
            Stopwatch sw = new();

            // Define a local function to perform the CPU work chunk
            void CPUWorkChunk(Tuple<int, int> partition)
            {
                int rows = input.GetLength(0);
                int cols = input.GetLength(1);
                int startRow = partition.Item1;
                int endRow = partition.Item2;
                for (int y = startRow; y < endRow; y++)
                {
                    for (int x = 0; x < cols; x++)
                    {
                        float sum = 0f;
                        // Sum neighbors
                        for (int i = -radius; i <= radius; i++)
                        {
                            for (int j = -radius; j <= radius; j++)
                            {
                                if (i == 0 && j == 0)
                                    continue; //Skip the center element
                                int nY = y + i, nX = x + j;
                                if (StaticUtils.IsInRange(nY, rows)
                                    && StaticUtils.IsInRange(nX, cols)
                                    )
                                {
                                    sum += input[nY, nX];
                                }
                            }
                        }
                        result[y, x] = sum;
                    }
                }
                return;
            }

            // Define GPU kernel as a local function to perform the work on GPU
            static void GPUWorkChung(ArrayView2D<float, Stride2D.DenseY> input, ArrayView2D<float, Stride2D.DenseY> output, int radius)
            {
                //How many threads is involved?
                int numOfGridThreads = Grid.DimX * Group.DimX;
                int currentThreadIdx = Grid.GlobalIndex.X;
                //Grid-Stride-Loop
                for (int myScopeIdx = currentThreadIdx; myScopeIdx < input.Length; myScopeIdx += numOfGridThreads)
                {
                    //Compute array indexes
                    int y = myScopeIdx / input.IntExtent.Y;
                    int x = myScopeIdx % input.IntExtent.Y;
                    //Sum neighbors around the center element
                    float sum = 0f;
                    for (int i = -radius; i <= radius; i++)
                    {
                        for (int j = -radius; j <= radius; j++)
                        {
                            if (i == 0 & j == 0)
                                continue; //Skip the center element
                            int nY = y + i;
                            int nX = x + j;
                            if (StaticUtils.IsInRange(nY, input.IntExtent.X)
                                && StaticUtils.IsInRange(nX, input.IntExtent.Y)
                                )
                            {
                                sum += input[nY, nX];
                            }
                        }
                    }
                    output[y, x] = sum;
                }
                return;
            }

            //Start measuring time
            sw.Start();

            //Try to acquire a GPU accelerator for processing
            GPUWrappedAccelerator? a = GPU.Allocator.Acquire();
            if (a == null)
            {
                // No GPU available, fallback to CPU processing
                Console.WriteLine($"[{threadName}] started processing on CPU...");
                //Process the input array in parallel using CPU
                Parallel.ForEach(Partitioner.Create(0, input.GetLength(0)), partition =>
                {
                    CPUWorkChunk(partition);
                });
            }
            else
            {
                // GPU is available, use it for processing
                Console.WriteLine($"[{threadName}] started processing on device {a.AccelObj.Device.Name}...");
                //Acquire a stream from the pool for GPU operations
                AcceleratorStream stream = a.StreamPool.Acquire();
                //Lock the accelerator to ensure thread safety
                lock (a.ThreadLock)
                {
                    //Allocate memory on the GPU for input and output arrays
                    using MemoryBuffer2D<float, Stride2D.DenseY> srcArray = a.AccelObj.Allocate2DDenseY<float>(stream, input);
                    using MemoryBuffer2D<float, Stride2D.DenseY> dstArray = a.AccelObj.Allocate2DDenseY<float>(srcArray.Extent);
                    //Load the GPU kernel using the cache on acquired GPUWrappedAccelerator instance
                    string kernelName = $"{nameof(TutorialApp)}.{nameof(AllocatorNDStreamExample)}.{nameof(NeighborSum)}.{nameof(GPUWorkChung)}";
                    var kernel = a.Kernels.GetOrAddKernel<Action<AcceleratorStream, KernelConfig, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int>>(
                                    kernelName,
                                    () => a.AccelObj.LoadKernel<ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int>(GPUWorkChung)
                                    );
                    //Execute the kernel with the grid and group dimensions set for grid-stride-loop
                    kernel(stream,
                           a.GetKernelConfig(srcArray.Length, true),
                           srcArray.View,
                           dstArray.View,
                           radius
                           );
                    stream.Synchronize();
                    dstArray.CopyToCPU(stream, result);
                }
                //Work is done, so release the stream back to the pool
                a.StreamPool.Release(stream);
                //Release back acquired GPU accelerator
                GPU.Allocator.Release(a);
            }
            //Stop measuring time
            sw.Stop();
            Console.WriteLine($"[{threadName}] finished in {sw.ElapsedMilliseconds} ms.");
            return result;
        }

        /// <summary>
        /// Executes a sequence of six parallel operations, each performing a neighbor sum calculation on specified 2D
        /// array of floating-point numbers.
        /// </summary>
        /// <remarks>This method utilizes <see cref="System.Threading.Tasks.Parallel.Invoke"/> to execute
        /// multiple neighbor sum calculations concurrently. The results of these calculations are returned as a tuple
        /// of six 2D arrays.</remarks>
        private (float[,], float[,], float[,], float[,], float[,], float[,]) ExecuteParallelSequence()
        {
            Stopwatch sw = new();
            Console.WriteLine();
            Console.WriteLine($"Parallel sequence started with {nameof(GPUAllocator)} operation mode {Enum.GetName(typeof(GPUAllocator.Mode), GPU.Allocator.CurrentMode())}...");
            float[,] resultT1 = null!, resultT2 = null!, resultT3 = null!, resultT4 = null!, resultT5 = null!, resultT6 = null!;
            // Start measuring time
            sw.Start();
            // Execute the neighbor sum calculations in parallel
            Parallel.Invoke(
                () => { resultT1 = NeighborSum($"T1", _2DArrayOfFloats1); },
                () => { resultT2 = NeighborSum($"T2", _2DArrayOfFloats2); },
                () => { resultT3 = NeighborSum($"T3", _2DArrayOfFloats3); },
                () => { resultT4 = NeighborSum($"T4", _2DArrayOfFloats4); },
                () => { resultT5 = NeighborSum($"T5", _2DArrayOfFloats5); },
                () => { resultT6 = NeighborSum($"T6", _2DArrayOfFloats6); }
                );
            // Stop measuring time
            sw.Stop();
            Console.WriteLine($"Parallel sequence completed in {sw.ElapsedMilliseconds} ms.");
            return (resultT1, resultT2, resultT3, resultT4, resultT5, resultT6);
        }

        /// <summary>
        /// Executes a series of parallel calculations using different GPU allocation modes.
        /// </summary>
        /// <remarks>This method sequentially sets the GPU allocation mode to various configurations 
        /// (NoAccelerator, MostPowerfulGPU, LeastPowerfulGPU and Standard) and executes parallel
        /// calculations for each mode.</remarks>
        public void Run()
        {
            //Set the GPU allocation mode to NoAccelerator to force CPU processing
            GPU.Allocator.SetMode(GPUAllocator.Mode.NoAccelerator);
            // Execute the parallel sequence
            _ = ExecuteParallelSequence();

            //Set the GPU allocation mode to MostPowerfulGPU to use the most powerful GPU available
            GPU.Allocator.SetMode(GPUAllocator.Mode.MostPowerfulGPU);
            // Execute the parallel sequence
            _ = ExecuteParallelSequence();

            //Set the GPU allocation mode to LeastPowerfulGPU to use the least powerful GPU available
            GPU.Allocator.SetMode(GPUAllocator.Mode.LeastPowerfulGPU);
            // Execute the parallel sequence
            _ = ExecuteParallelSequence();

            //Set the GPU allocation mode to Standard to use the default GPU allocation mode where all GPU devices are used
            GPU.Allocator.SetMode(GPUAllocator.Mode.Standard);
            // Execute the parallel sequence
            _ = ExecuteParallelSequence();

            return;
        }

    }
}
