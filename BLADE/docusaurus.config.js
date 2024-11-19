// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

const config = {
  title: 'BLADE',
  tagline: 'Bayesian Log Normal Deconvolution',
  favicon: 'img/favicon.ico',
  url: 'https://your-docusaurus-site.example.com', // Replace with your actual site URL
  baseUrl: '/',
  organizationName: 'your-org-name', // Replace with your GitHub organization/user name
  projectName: 'blade', // Replace with your GitHub repository name

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org-name/blade/edit/main/', // Link to edit docs in GitHub
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/your-org-name/blade/edit/main/', // Link to edit blog posts in GitHub
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/logo_final_small.png', // Image used for social media previews
    navbar: {
      title: 'BLADE',
      logo: {
        alt: 'BLADE Logo',
        src: 'img/logo_final_small.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'installation',
          position: 'left',
          label: 'Installation',
        },
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Tutorial',
        },
        {
          type: 'doc',
          docId: 'about-us',
          position: 'left',
          label: 'About Us',
        },
        { to: '/blog', label: 'Blog', position: 'left' },
        {
          href: 'https://github.com/tgac-vumc/BLADE',
          label: 'GitHub',
          position: 'right',
        },
        // Placeholder for the search button (to be implemented in CustomNavbar)
      ],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
  },
};

export default config;
