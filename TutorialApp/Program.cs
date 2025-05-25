// See https://aka.ms/new-console-template for more information


using PPU4ILGPU;
using TutorialApp;

bool exit = false;
while (!exit)
{
    Console.WriteLine("Menu");
    Console.WriteLine($"1. {nameof(GPUWrappedAccelerator)} used as a standalone component");
    Console.WriteLine($"2. {nameof(GPUAllocator)} singleton scenario");
    Console.WriteLine($"3. {nameof(GPUStreamPool)} usage of non-default GPU streams");
    Console.WriteLine("4. Exit");
    Console.WriteLine("Press your choice...");
    ConsoleKeyInfo keyInfo = Console.ReadKey();
    switch (keyInfo.KeyChar)
    {
        case '1':
            {
                using AcceleratorExample example = new();
                example.Run();
            }
            break;
        case '2':
            {
                AllocatorExample example = new();
                example.Run();
            }
            break;
        case '3':
            {
                AllocatorNDStreamExample example = new();
                example.Run();
            }
            break;
        case '4':
            Console.WriteLine("Exiting...");
            exit = true;
            break;
        default:
            break;
    }
    Console.Clear();
}
