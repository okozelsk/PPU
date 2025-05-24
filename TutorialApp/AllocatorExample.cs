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
    public class AllocatorExample
    {
        private readonly float[,] _big2DArrayOfFloats;
        private readonly float[,] _small2DArrayOfFloats;

        /// <summary>
        /// Initializes a new instance of the <see cref="AllocatorExample"/> class.
        /// </summary>
        /// <remarks>This constructor ensures the GPUAllocator singleton is initialized.</remarks>
        public AllocatorExample()
        {
            Console.Clear();
            Console.WriteLine("Ensuring the singleton instance of GPUAllocator is initialized...");
            GPU.Allocator.SetMode(GPUAllocator.Mode.Standard);
            Random rand = new();
            Console.WriteLine("Initializing big 2D array of random float values for example purposes...");
            _big2DArrayOfFloats = new float[1280, 5120];
            for (int i = 0; i < _big2DArrayOfFloats.GetLength(0); i++)
            {
                for (int j = 0; j < _big2DArrayOfFloats.GetLength(1); j++)
                {
                    _big2DArrayOfFloats[i, j] = (float)rand.NextDouble();
                }
            }
            Console.WriteLine("Initializing small 2D array of random float values for example purposes...");
            _small2DArrayOfFloats = new float[128, 512];
            for (int i = 0; i < _small2DArrayOfFloats.GetLength(0); i++)
            {
                for (int j = 0; j < _small2DArrayOfFloats.GetLength(1); j++)
                {
                    _small2DArrayOfFloats[i, j] = (float)rand.NextDouble();
                }
            }
            Console.WriteLine();
        }


        /// <summary>
        /// Sums all neighbors around each element in 2D array.
        /// Sums are stored and returned in 2D float array.
        /// </summary>
        private float[,] NeighborSum(string threadName, float[,] input)
        {
            float[,] result = new float[input.GetLength(0), input.GetLength(1)];
            
            void CPUWorkChunk(Tuple<int, int>partition)
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
                        for (int i = -1; i <= 1; i++)
                        {
                            for (int j = -1; j <= 1; j++)
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

            static void GPUWorkChung(ArrayView2D<float, Stride2D.DenseY> input, ArrayView2D<float, Stride2D.DenseY> output)
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
                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
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

            GPUWrappedAccelerator? a = GPU.Allocator.Acquire();
            if (a == null)
            {
                Console.WriteLine($"[{threadName}] started processing on CPU...");
                Parallel.ForEach(Partitioner.Create(0, input.GetLength(0)), partition =>
                {
                    CPUWorkChunk(partition);
                });
            }
            else
            {
                Console.WriteLine($"[{threadName}] started processing on device {a.AccelObj.Device.Name}...");
                lock (a.ThreadLock)
                {
                    using MemoryBuffer2D<float, Stride2D.DenseY> srcArray = a.AccelObj.Allocate2DDenseY<float>(input);
                    using MemoryBuffer2D<float, Stride2D.DenseY> dstArray = a.AccelObj.Allocate2DDenseY<float>(srcArray.IntExtent);
                    string kernelName = $"{nameof(TutorialApp)}.{nameof(AllocatorExample)}.{nameof(NeighborSum)}.{nameof(GPUWorkChung)}";
                    var kernel = a.Kernels.GetOrAddKernel<Action<KernelConfig, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>>>(
                                    kernelName,
                                    () => a.AccelObj.LoadStreamKernel<ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>>(GPUWorkChung)
                                    );
                    ///////////////////////////////////////////////////////
                    //Execute the kernel with the specified grid and group dimensions.
                    kernel(a.GetKernelConfig(srcArray.Length, true),
                           srcArray.View,
                           dstArray.View
                           );
                    a.AccelObj.Synchronize();
                    dstArray.CopyToCPU(result);
                }
            }
            Console.WriteLine($"[{threadName}] finished...");
            return result;
        }

        private void ExecuteParallelSequence()
        {
            Stopwatch sw = new Stopwatch();
            Console.WriteLine($"Parallel sequence started with {nameof(GPUAllocator)} operation mode {Enum.GetName(typeof(GPUAllocator.Mode), GPU.Allocator.CurrentMode())}...");
            sw.Start();
            Parallel.Invoke(
                () => { _ = NeighborSum("T1 big", _big2DArrayOfFloats); },
                () => { _ = NeighborSum("T2 small", _small2DArrayOfFloats); },
                () => { _ = NeighborSum("T3 small", _small2DArrayOfFloats); },
                () => { _ = NeighborSum("T4 small", _small2DArrayOfFloats); },
                () => { _ = NeighborSum("T5 small", _small2DArrayOfFloats); },
                () => { _ = NeighborSum("T6 small", _small2DArrayOfFloats); }
                );
            sw.Stop();
            Console.WriteLine($"Parallel sequence completed in {sw.ElapsedMilliseconds} ms.");
            return;
        }


        public void Run()
        {
            GPU.Allocator.SetMode(GPUAllocator.Mode.CPUAccelerator);
            ExecuteParallelSequence();
            GPU.Allocator.SetMode(GPUAllocator.Mode.NoAccelerator);
            ExecuteParallelSequence();
            GPU.Allocator.SetMode(GPUAllocator.Mode.MostPowerfulGPU);
            ExecuteParallelSequence();
            GPU.Allocator.SetMode(GPUAllocator.Mode.LeastPowerfulGPU);
            ExecuteParallelSequence();
            GPU.Allocator.SetMode(GPUAllocator.Mode.Standard);
            ExecuteParallelSequence();

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return;
        }

    }
}
