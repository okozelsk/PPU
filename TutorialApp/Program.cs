// See https://aka.ms/new-console-template for more information


using PPU4ILGPU;
using TutorialApp;

bool exit = false;
while (!exit)
{
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine("Main menu");
    Console.WriteLine("---------");
    Console.WriteLine($"1. {nameof(GPUWrappedAccelerator)} used as a standalone component");
    Console.WriteLine($"2. {nameof(GPUAllocator)} singleton scenario");
    Console.WriteLine($"3. {nameof(GPUStreamPool)} usage of non-default GPU streams");
    Console.WriteLine($"4. {nameof(GPUAllocator)} reservation of {nameof(GPUWrappedAccelerator)} for exclusive use");
    Console.WriteLine("5. Exit");
    Console.WriteLine("Select your choice (press key 1, 2, 3, 4 or 5)...");
    ConsoleKeyInfo keyInfo = Console.ReadKey();
    Console.WriteLine();
    Console.WriteLine();
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
            {
                AllocatorGPUReservationExample example = new();
                example.Run();
            }
            break;
        case '5':
            Console.WriteLine("Exiting...");
            exit = true;
            break;
        default:
            Console.WriteLine("Unsupported choice.");
            break;
    }
    Console.WriteLine();
    Console.WriteLine();
    if (!exit)
    {
        Console.WriteLine("Press any key to return to main menu...");
        Console.ReadKey();
    }
}
