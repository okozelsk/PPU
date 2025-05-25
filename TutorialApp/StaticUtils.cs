using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TutorialApp
{
    /// <summary>
    /// Provides static utility methods.
    /// </summary>
    public static class StaticUtils
    {
        /// <summary>
        /// Determines whether the specified index value is within the valid range for a given dimension.
        /// </summary>
        /// <param name="idx">The index value to check.</param>
        /// <param name="dim">The upper bound of the range (exclusive). Must be greater than or equal to 0.</param>
        /// <returns><see langword="true"/> if <paramref name="idx"/> is greater than or equal to 0 and less than <paramref
        /// name="dim"/>; otherwise, <see langword="false"/>.</returns>
        public static bool IsInRange(int idx, int dim)
        {
            return idx >= 0 && idx < dim;
        }

    }
}
