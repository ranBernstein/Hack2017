using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.IdentityModel.Clients.ActiveDirectory;

namespace ConsoleApplication1
{
    class Program
    {
        static void Main(string[] args)
        {
            var loginUrl = "https://login.microsoftonline.com/microsoft.onmicrosoft.com";
            var resource = "https://graph.microsoft.com";

            var authContext = new Microsoft.IdentityModel.Clients.ActiveDirectory.AuthenticationContext(loginUrl);

            const string ClientId = "e81daf8f-2b30-43d0-b6a9-737dc3353126";
            const string RedirectUrl = "http://localhost/";

            // Note that there are other options for PromptBehavior so that it will not cache credentials, etc
            var authResult = authContext.AcquireTokenAsync(resource, ClientId, new Uri(RedirectUrl), new PlatformParameters(PromptBehavior.Auto));
            Console.WriteLine(authResult.Result.AccessToken);
            //Console.ReadLine();
        }
    }
}
