// See https://aka.ms/new-console-template for more information


using TutorialApp;

bool exit = false;
while (!exit)
{
    Console.WriteLine("Menu");
    Console.WriteLine("1. GPUWrappedAccelerator with GPUKernelCache versus classical ILGPU LoadKernel approach");
    Console.WriteLine("2. GPUAllocator usage");
    Console.WriteLine("3. GPUStreamPool option");
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
            break;
        case '3':
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
