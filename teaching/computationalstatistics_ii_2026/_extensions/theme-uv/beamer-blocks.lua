-- Only the collapsible shell needs raw HTML; its title is always plain text.
local function escape_html(text)
  return text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
    :gsub('"', '&quot;'):gsub("'", '&#39;')
end

function Div(el)
  if el.classes:includes('eqbox') then
    local title = el.attributes['title']
    local colour = el.attributes['colour'] or el.attributes['color']
    local alpha = el.attributes['alpha']
    local scroll = el.attributes['scroll']
    if scroll and scroll ~= 'true' and scroll ~= 'false' then
      error('eqbox: scroll must be true or false.')
    end
    if scroll == 'true' then el.classes:insert('eqbox-scroll') end
    el.attributes['scroll'] = nil
    local styles = {}
    if colour then
      local palette = {blue = '2d73e1', red = 'be2832', purple = '7d3c98'}
      local hex = palette[colour:lower()] or colour:match('^#(%x%x%x%x%x%x)$')
      if not hex then
        error('eqbox: colour must be blue, red, purple, or a six-digit hex colour such as #0F4494.')
      end
      styles[#styles + 1] = string.format('--eqbox-rgb: %d, %d, %d;',
        tonumber(hex:sub(1, 2), 16), tonumber(hex:sub(3, 4), 16),
        tonumber(hex:sub(5, 6), 16))
    end
    if alpha then
      local value = tonumber(alpha)
      if not value or value ~= value or value < 0 or value > 1 then
        error('eqbox: alpha must be a number between 0 and 1.')
      end
      styles[#styles + 1] = '--eqbox-alpha: ' .. tostring(value) .. ';'
    end
    if #styles > 0 then
      el.attributes['style'] = (el.attributes['style'] or '') .. ';' .. table.concat(styles, ' ')
    end
    el.attributes['colour'] = nil
    el.attributes['color'] = nil
    el.attributes['alpha'] = nil
    el.content = pandoc.Blocks({pandoc.Div(
      el.content, pandoc.Attr('', {'eqbox-body'})
    )})
    if title and title ~= '' then
      el.content:insert(1, pandoc.Div(
        {pandoc.Plain({pandoc.Str(title)})},
        pandoc.Attr('', {'eqbox-title'})
      ))
      el.attributes['title'] = nil
    end
    return el
  end

  if el.classes:includes('proof_idea') or el.classes:includes('proofbox') then
    -- Proof and proof idea are variants of the same collapsible component.
    local is_idea = el.classes:includes('proof_idea')
    local box_type = is_idea and 'idea' or 'proof'
    local title = el.attributes['title'] or (is_idea and 'Proof Idea' or 'Proof')
    local open_tag = string.format(
      '<details class="beamer-box %s">\n<summary>%s</summary>\n\n',
      box_type, escape_html(title)
    )
    local result = pandoc.List({pandoc.RawBlock('html', open_tag)})
    -- Keep the body as Pandoc blocks so mathematics and nested content still render.
    result:extend(el.content)
    result:insert(pandoc.RawBlock('html', '\n\n</details>'))
    -- Keep the labelled outer div so proofs can be linked to and recalled.
    el.content = result
    el.attributes['title'] = nil
    return el
    
  elseif el.classes:includes('box') then
    local title = el.attributes['title']
    if title and title:match('%S') then
      -- A plain-text title: Pandoc escapes HTML while retaining the content below.
      el.content:insert(1, pandoc.Div(
        {pandoc.Plain({pandoc.Str(title)})},
        pandoc.Attr('', {'box-title'})
      ))
    end
    el.attributes['title'] = nil
    return el
  end
end
