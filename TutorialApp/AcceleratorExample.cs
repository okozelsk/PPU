using ILGPU;
using ILGPU.Backends.PTX;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using PPU4ILGPU;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TutorialApp
{
    public class AcceleratorExample : IDisposable
    {
        private readonly Context _context;
        private readonly Device _device;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="AcceleratorExample"/> class.
        /// </summary>
        /// <remarks>This constructor sets up the necessary ILGPU context and selects the preferred GPU
        /// device. It also initializes the internal state to indicate that the instance has not been
        /// disposed.</remarks>
        public AcceleratorExample()
        {
            Console.Clear();
            Console.WriteLine("PPU4ILGPU example GPUWrappedAccelerator pattern started");
            Console.WriteLine("Initializing ILGPU context...");
            _context = Context.CreateDefault();
            Console.WriteLine("Context object obtained");
            Console.WriteLine($"Asking preferred device...");
            _device = _context.GetPreferredDevice(false);
            Console.WriteLine($"Preferred device {_device.Name} obtained");
            _isDisposed = false;
            Console.WriteLine();
        }

        /// <summary>
        /// Determines whether the specified value is within the valid range for a given dimension.
        /// </summary>
        /// <param name="d">The value to check.</param>
        /// <param name="dim">The upper bound of the range (exclusive). Must be greater than or equal to 0.</param>
        /// <returns><see langword="true"/> if <paramref name="d"/> is greater than or equal to 0 and less than <paramref
        /// name="dim"/>; otherwise, <see langword="false"/>.</returns>
        private static bool IsInRange(int d, int dim)
        {
            return d >= 0 && d < dim;
        }

        /// <summary>
        /// Nothing special. Performs a grid-stride loop operation on a 2D source array, applying a neighborhood-based computation and
        /// storing the results in a 2D destination array.
        /// </summary>
        /// <param name="srcArray">The 2D source array of bytes. Each element represents the input data for the computation.</param>
        /// <param name="dstArray">The 2D destination array of floats. The computed results are stored in this array.</param>
        private static void GPUWorkChung(ArrayView2D<byte, Stride2D.DenseY> srcArray,
                                         ArrayView2D<float, Stride2D.DenseY> dstArray
                                         )
        {
            //How many threads is involved?
            int numOfGridThreads = Grid.DimX * Group.DimX;
            int currentThreadIdx = Grid.GlobalIndex.X;
            //Grid-Stride-Loop
            for (int myScopeIdx = currentThreadIdx; myScopeIdx < srcArray.Length; myScopeIdx += numOfGridThreads)
            {
                //Compute array indexes
                int y = myScopeIdx / srcArray.IntExtent.Y;
                int x = myScopeIdx % srcArray.IntExtent.Y;
                //Do the work
                for (int i = -1; i <= 1; i++)
                {
                    for(int j = -1; j <= 1; j++)
                    {
                        if (IsInRange(y + i, srcArray.IntExtent.X) && IsInRange(x + j, srcArray.IntExtent.Y))
                        {
                            dstArray[y, x] += srcArray[y + i, x + j];
                        }
                    }
                }
                dstArray[y, x] *= 2.0f;
            }
            return;
        }

        /// <summary>
        /// This method uses a classical ILGPU computation pattern, which relies on the ILGPU
        /// internal cache mechanism to load the kernel efficiently.
        /// </summary>
        private static float[,] DoClassicalPattern(Accelerator a, Index2D dataSize)
        {
            float[,] result = new float[dataSize.X, dataSize.Y];
            using MemoryBuffer2D<byte, Stride2D.DenseY> srcArray = a.Allocate2DDenseY<byte>(dataSize);
            using MemoryBuffer2D<float, Stride2D.DenseY> dstArray = a.Allocate2DDenseY<float>(dataSize);
            srcArray.MemSet(128);

            //////////////////////////////////////////////////////
            // The Classical pattern means to load kernel when needed relying on ILGPU internal cache mechanism.
            // Unfortunately there must be an internal bug, because in the most situations, already
            // known and compiled kernel is probably compiled again and again.
            var kernel = a.LoadStreamKernel<ArrayView2D<byte, Stride2D.DenseY> , ArrayView2D<float, Stride2D.DenseY>>(GPUWorkChung);
            //////////////////////////////////////////////////////

            //Execute the kernel with the specified grid and group dimensions.
            kernel(new KernelConfig(a.MaxNumGroupsExtent.Item1, a.MaxNumGroupsExtent.Item2),
                    srcArray.View,
                    dstArray.View
                    );
            a.Synchronize();
            dstArray.CopyToCPU(result);
            return result;
        }

        /// <summary>
        /// This method does exactly the same as DoClassicalPattern method
        /// using GPUWrappedAccelerator instead of Accelerator.
        /// The difference is only in utilizing the GPUWrappedAccelerator's
        /// pinned cache of already compiled kernels, to prevent a probable
        /// bug in the ILGPU internal cache mechanism.
        private static float[,] DoGPUWrappedAcceleratorPattern(GPUWrappedAccelerator a, Index2D dataSize)
        {
            float[,] result = new float[dataSize.X, dataSize.Y];
            using MemoryBuffer2D<byte, Stride2D.DenseY> srcArray = a.AccelObj.Allocate2DDenseY<byte>(dataSize);
            using MemoryBuffer2D<float, Stride2D.DenseY> dstArray = a.AccelObj.Allocate2DDenseY<float>(dataSize);
            srcArray.MemSet(128);

            //////////////////////////////////////////////////////
            // The GPUWrappedAccelerator pattern means to load compiled kernel when needed
            // using the accelerator's pinned cache of already compiled named kernels.
            // Cache is implemented as thread safe dictionary mapping kernel name to delegate.
            // So kernel name is used to identify the kernel in the cache and is up to you how
            // kernel name is constructed.
            // Good practice is to use full path to the kernel's code like here.
            string kernelName = "TutorialApp.AcceleratorExample.GPUWorkChung";
            var kernel = a.Kernels.GetOrAddKernel<Action<KernelConfig, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>>>(
                            kernelName,
                            () => a.AccelObj.LoadStreamKernel<ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>>(GPUWorkChung)
                            );
            ///////////////////////////////////////////////////////

            //Execute the kernel with the specified grid and group dimensions.
            kernel(a.GetKernelConfig(srcArray.Length, true),
                   srcArray.View,
                   dstArray.View
                   );
            a.AccelObj.Synchronize();
            dstArray.CopyToCPU(result);
            return result;
        }

        /// <summary>
        /// Executes performance tests comparing the Classical pattern and the GPUWrappedAccelerator pattern.
        /// </summary>
        /// <remarks>This method performs a series of tests to measure and compare the execution times of
        /// two "load kernel" patterns: the Classical pattern and the GPUWrappedAccelerator pattern. Each
        /// pattern is executed a fixed number of times using exactly the same kernel, and the elapsed
        /// time for each is recorded and displayed.</remarks>
        public void Run()
        {
            Stopwatch sw = new Stopwatch();
            int repetitions = 1000;
            Index2D dataSize = new Index2D(128, 512);

            void PerformClassicalPatternTest()
            {
                using Accelerator accelerator = _device.CreateAccelerator(_context);
                for (int i = 0; i < repetitions; i++)
                {
                    _ = DoClassicalPattern(accelerator, dataSize);
                }
            }

            void PerformGPUWrappedAcceleratorPatternTest()
            {
                using GPUWrappedAccelerator accelerator = new(_context, _device);
                for (int i = 0; i < repetitions; i++)
                {
                    _ = DoGPUWrappedAcceleratorPattern(accelerator, dataSize);
                }
            }

            Console.WriteLine("Test of the Classical pattern started...");
            sw.Reset();
            sw.Start();
            PerformClassicalPatternTest();
            sw.Stop();
            long classicalPatternTime = sw.ElapsedMilliseconds;
            Console.WriteLine($"    Test run took {classicalPatternTime} ms.");
            
            Console.WriteLine("Test of the GPUWrappedAccelerator pattern started...");
            sw.Reset();
            sw.Start();
            PerformGPUWrappedAcceleratorPatternTest();
            sw.Stop();
            long gpuWrappedAcceleratorPatternTime = sw.ElapsedMilliseconds;
            Console.WriteLine($"    Test run took {gpuWrappedAcceleratorPatternTime} ms.");
            
            double ratio = Math.Round(classicalPatternTime >= gpuWrappedAcceleratorPatternTime ?
                                      (double)classicalPatternTime / (double)gpuWrappedAcceleratorPatternTime
                                      : (double)gpuWrappedAcceleratorPatternTime / (double)classicalPatternTime,
                                      1,
                                      MidpointRounding.AwayFromZero
                                      );
            string result = classicalPatternTime >= gpuWrappedAcceleratorPatternTime ? "faster" : "slower";
            Console.WriteLine($"Results is that the GPUWrappedAccelerator pattern is about {ratio} times {result} than the Classical pattern.");

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
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
            if (!_isDisposed)
            {
                _isDisposed = true;
                if (disposing)
                {
                    Console.WriteLine("Disposing ILGPU context...");
                    _context.Dispose();
                }
            }
            return;
        }
    }
}
