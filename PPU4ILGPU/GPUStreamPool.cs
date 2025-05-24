using ILGPU.Runtime;
using System.Collections.Concurrent;

namespace PPU4ILGPU
{
    /// <summary>
    /// Manages a pool of reusable GPU streams for a specific accelerator, allowing efficient acquisition and release of
    /// streams to minimize resource creation overhead.
    /// </summary>
    /// <remarks>This class is designed to manage GPU streams for an accelerator by maintaining a pool of
    /// reusable streams. Streams can be acquired using the <see cref="Acquire"/> method and released back to the pool
    /// using the <see cref="Release"/> method. If no streams are available in the pool, a new stream is created. The
    /// pool ensures that streams are properly disposed when the <see cref="Dispose"/> method is called.  This class is
    /// thread-safe and can be used in concurrent scenarios.</remarks>
    public sealed class GPUStreamPool : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly ConcurrentBag<AcceleratorStream> _availableStreams;
        private readonly List<AcceleratorStream> _createdStreams;
        private readonly object _syncRoot;
        private int _isDisposed;

        /// <summary>
        /// Initializes a new instance of the <see cref="GPUStreamPool"/> class,  which manages a pool of GPU streams
        /// for efficient reuse.
        /// </summary>
        /// <param name="accelerator">The <see cref="Accelerator"/> instance associated with the GPU streams.  This parameter cannot be <see
        /// langword="null"/>.</param>
        /// <exception cref="ArgumentNullException"></exception>
        public GPUStreamPool(Accelerator accelerator)
        {
            ArgumentNullException.ThrowIfNull(accelerator);
            _accelerator = accelerator;
            _availableStreams = new();
            _createdStreams = new();
            _syncRoot = new object();
            _isDisposed = 0;
            return;
        }

        /// <summary>
        /// Acquires an available <see cref="AcceleratorStream"/> from the pool or creates a new one if none are
        /// available.
        /// </summary>
        /// <remarks>If a stream is available in the pool, it is returned. Otherwise, a new <see
        /// cref="AcceleratorStream"/> is created and added to the internal collection of created streams.</remarks>
        /// <returns>An <see cref="AcceleratorStream"/> instance that can be used for operations.</returns>
        /// <exception cref="ObjectDisposedException"></exception>
        public AcceleratorStream Acquire()
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            if (_availableStreams.TryTake(out AcceleratorStream? stream))
            {
                return stream;
            }
            else
            {
                stream = _accelerator.CreateStream();
                lock (_syncRoot)
                {
                    _createdStreams.Add(stream);
                }
                return stream;
            }
        }

        /// <summary>
        /// Releases the specified accelerator stream back to the pool of available streams.
        /// </summary>
        /// <param name="stream">The <see cref="AcceleratorStream"/> to release. Cannot be <see langword="null"/>.</param>
        /// <exception cref="ObjectDisposedException"></exception>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="ArgumentNullException"></exception>
        public void Release(AcceleratorStream stream)
        {
            ObjectDisposedException.ThrowIf(_isDisposed != 0, this);
            ArgumentNullException.ThrowIfNull(stream);
            if (stream.Accelerator != _accelerator)
            {
                throw new ArgumentException("The stream does not belong to this accelerator.", nameof(stream));
            }
            _availableStreams.Add(stream);
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
                        foreach (var stream in _createdStreams)
                        {
                            stream.Dispose();
                        }
                        _createdStreams.Clear();
                    }
                }
            }
            return;
        }
    }
}
